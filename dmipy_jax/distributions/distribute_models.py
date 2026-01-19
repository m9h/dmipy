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
    parameter_cardinality: dict = eqx.field(static=True)
    parameter_ranges: dict = eqx.field(static=True)
    
    def __init__(self, base_model, distribution, target_parameter='diameter'):
        self.base_model = base_model
        self.distribution = distribution
        self.target_parameter = target_parameter
        
        # Combine parameter names/ranges
        # Base model params: replace target_parameter with distribution params
        p_names = list(base_model.parameter_names)
        if target_parameter in p_names:
            p_names.remove(target_parameter)
        
        p_names.extend(distribution.parameter_names)
        self.parameter_names = tuple(p_names)
        
        self.parameter_cardinality = base_model.parameter_cardinality.copy()
        if target_parameter in self.parameter_cardinality:
            del self.parameter_cardinality[target_parameter]
        self.parameter_cardinality.update(distribution.parameter_cardinality)
        
        self.parameter_ranges = base_model.parameter_ranges.copy()
        if target_parameter in self.parameter_ranges:
            del self.parameter_ranges[target_parameter]
        self.parameter_ranges.update(distribution.parameter_ranges)

    def __call__(self, bvals, bvecs, **kwargs):
        # 1. Get distribution grid
        dist_kwargs = {k: kwargs[k] for k in self.distribution.parameter_names if k in kwargs}
        # Use default if not present (handled by distribution's __call__ if it uses self.param)
        # But our models usually expect kwargs passed fully.
        # For Equinox models, we might need to rely on passing explicit params if stateless, 
        # or self.params if stateful. Dmipy-JAX models seem to use kwargs.
        
        domain, probability = self.distribution(**dist_kwargs)
        
        # 2. Integrate
        # We scan/vmap over the domain (e.g. diameters)
        
        def evaluate_at_point(point_val):
            # Update kwargs with the target parameter value
            point_kwargs = kwargs.copy()
            point_kwargs[self.target_parameter] = point_val
            
            # Base model call
            return self.base_model(bvals, bvecs, **point_kwargs)
            
        # Vmap over domain
        signals = jax.vmap(evaluate_at_point)(domain) # (N_steps, N_measurements)
        
        # Integrate: integral( S(x) * P(x) dx )
        # weighted_signals = signals * probability[:, None]
        # result = jnp.trapz(weighted_signals, x=domain, axis=0)
        
        # Better: P(x) is PDF. S_total = Integral S(x) P(x) dx.
        weighted_signals = signals * probability[:, None]
        # Use trapezoid if available, else manual
        if hasattr(jnp, 'trapezoid'):
            result = jnp.trapezoid(weighted_signals, x=domain, axis=0)
            pdf_area = jnp.trapezoid(probability, x=domain)
        else:
            # Fallback for trapz if it exists, roughly equivalent
            result = jnp.trapz(weighted_signals, x=domain, axis=0)
            pdf_area = jnp.trapz(probability, x=domain)
        
        result = result / pdf_area
        
        return result
