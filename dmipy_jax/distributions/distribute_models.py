import jax
import jax.numpy as jnp
import equinox as eqx

__all__ = ['DistributedModel']

class DistributedModel(eqx.Module):
    """
    Integrates a base model over a distribution of a target parameter.
    
    e.g. RestrictedCylinder distributed over 'diameter' using DD1Gamma.
    """
    base_model: eqx.Module
    distribution: eqx.Module
    target_parameter: str = eqx.field(static=True)
    parameter_names: tuple = eqx.field(static=True)
    _parameter_cardinality: tuple = eqx.field(static=True)
    _parameter_ranges: tuple = eqx.field(static=True)
    
    def __init__(self, base_model, distribution, target_parameter='diameter'):
        self.base_model = base_model
        self.distribution = distribution
        self.target_parameter = target_parameter
        
        # Combine parameter names/ranges
        p_names = list(base_model.parameter_names)
        if target_parameter in p_names:
            p_names.remove(target_parameter)
        
        p_names.extend(distribution.parameter_names)
        self.parameter_names = tuple(p_names)
        
        card = base_model.parameter_cardinality.copy()
        if target_parameter in card:
            del card[target_parameter]
        card.update(distribution.parameter_cardinality)
        self._parameter_cardinality = tuple(sorted(card.items()))
        
        ranges = base_model.parameter_ranges.copy()
        if target_parameter in ranges:
            del ranges[target_parameter]
        ranges.update(distribution.parameter_ranges)
        
        # Ranges values are mutable (lists), need to make them hashable (tuples)
        def to_hashable(val):
            if isinstance(val, list):
                return tuple(to_hashable(v) for v in val)
            if isinstance(val, tuple):
                return tuple(to_hashable(v) for v in val)
            if isinstance(val, dict):
                return tuple(sorted((k, to_hashable(v)) for k, v in val.items()))
            return val

        self._parameter_ranges = tuple(sorted((k, to_hashable(v)) for k, v in ranges.items()))

    @property
    def parameter_cardinality(self):
        return dict(self._parameter_cardinality)

    @property
    def parameter_ranges(self):
        # We need to reconstruct the original structure roughly? 
        # Or just dict is enough. But values might be tuples now instead of lists.
        # This should be fine for reading.
        return dict(self._parameter_ranges)

    def __call__(self, bvals, gradient_directions, **kwargs):
        # 1. Get distribution grid
        dist_kwargs = {k: kwargs[k] for k in self.distribution.parameter_names if k in kwargs}
        # Use default if not present (handled by distribution's __call__ if it uses self.param)
        # But our models usually expect kwargs passed fully.
        # For Equinox models, we might need to rely on passing explicit params if stateless, 
        # or self.params if stateful. Dmipy-JAX models seem to use kwargs.
        
        domain, probability = self.distribution(**dist_kwargs)
        
        # 2. Integrate
        # We scan/vmap over the domain
        
        def evaluate_at_point(point_val):
            # Update kwargs with the target parameter value
            point_kwargs = kwargs.copy()
            point_kwargs[self.target_parameter] = point_val
            
            # Base model call
            # Ensure base model receives gradient_directions
            return self.base_model(bvals, gradient_directions=gradient_directions, **point_kwargs)
            
        # Vmap over domain
        signals = jax.vmap(evaluate_at_point)(domain) # (N_steps, N_measurements)
        
        # Integrate: integral( S(x) * P(x) dx ) or sum( S(x) * w(x) )
        weighted_signals = signals * probability[:, None]
        
        # Check if domain is likely 1D continuous (scalar per point) or multidimensional (vector per point)
        # If domain is (N,) -> 1D continuous -> Trapezoid
        # If domain is (N, D) -> Multi-D (Spherical) -> Weighted Sum (assuming prob are weights)
        
        if domain.ndim == 1:
             # Use trapezoid if available, else manual
            if hasattr(jnp, 'trapezoid'):
                result = jnp.trapezoid(weighted_signals, x=domain, axis=0)
                pdf_area = jnp.trapezoid(probability, x=domain)
            else:
                # Fallback for trapz if it exists, roughly equivalent
                result = jnp.trapz(weighted_signals, x=domain, axis=0)
                pdf_area = jnp.trapz(probability, x=domain)
            
            # Safe division
            pdf_area = jnp.where(pdf_area < 1e-12, 1.0, pdf_area)
            result = result / pdf_area
        else:
            # Assume Discrete / Weighted Sum
            # probability contains the weights sum(w) should be 1.
            result = jnp.sum(weighted_signals, axis=0)
            # No normalization needed if weights already sum to 1?
            # Or we can normalize if we want robustness.
            # let's assume probability are weights.
            pass
        
        return result
