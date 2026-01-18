
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Sequence, Callable, Dict, Any, Tuple
from jaxtyping import Array, Float
from dataclasses import dataclass
from dmipy_jax.core.surrogate import PolynomialChaosExpansion

class Distribution(eqx.Module):
    """Abstract base class for parameter distributions."""
    pass

class Uniform(Distribution):
    """Uniform distribution U(low, high)."""
    low: float
    high: float
    
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        
    def name(self) -> str:
        return 'Uniform'

class Normal(Distribution):
    """Normal distribution N(loc, scale)."""
    loc: float
    scale: float
    
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale
        
    def name(self) -> str:
        return 'Normal'

def propagate_uncertainty(
    model_func: Callable,
    fixed_params: Dict[str, Any],
    acquisition: Any,
    parameter_names: Sequence[str],
    uncertain_params: Dict[str, Distribution],
    order: int = 3,
    num_samples: int = 200 # For fitting the surrogate
) -> Tuple[Float[Array, " num_measurements"], Float[Array, " num_measurements"]]:
    """
    Propagates uncertainty from input parameters to model output using gPC.
    
    Args:
        model_func: Function f(params_array, acquisition) -> signal
        fixed_params: Dictionary of fixed parameter values (scalars).
        acquisition: Acquisition object.
        parameter_names: List of all parameter names expected by the model in order.
        uncertain_params: Dictionary mapping parameter names to Distribution objects.
        order: gPC expansion order.
        num_samples: Number of samples to train the surrogate.
        
    Returns:
        (mean_signal, std_signal)
    """
    
    # 1. Identify dimensions and distributions
    dist_names = []
    dist_objects = []
    uncertain_names = list(uncertain_params.keys())
    
    for name in uncertain_names:
        dist = uncertain_params[name]
        dist_names.append(dist.name())
        dist_objects.append(dist)
        
    num_dims = len(uncertain_names)
    
    # 2. Generate Training Samples from Distributions
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_dims)
    
    samples_list = []
    
    # We need to map standard distributions (Legendre/Hermite domain) to physical domain
    # Legendre: [-1, 1] -> Uniform[low, high]
    # Hermite: N(0, 1) -> Normal[loc, scale]
    
    # We generate samples in the TRANSFORMED (physical) domain to run the physical model.
    # BUT the gPC fit expects inputs in the STANDARD domain [-1, 1] or N(0,1).
    
    # So we generate standard samples X_std, then transform to X_physical.
    
    X_std_list = []
    X_phys_list = []
    
    for i, dist in enumerate(dist_objects):
        k = keys[i]
        if isinstance(dist, Uniform):
            # Standard: U[-1, 1]
            x_std = jax.random.uniform(k, (num_samples, 1), minval=-1.0, maxval=1.0)
            
            # Physical: Map [-1, 1] to [low, high]
            # x_phys = low + (x_std + 1) * 0.5 * (high - low)
            mid = (dist.low + dist.high) / 2.0
            width = (dist.high - dist.low) / 2.0
            x_phys = mid + x_std * width
            
        elif isinstance(dist, Normal):
            # Standard: N(0, 1)
            x_std = jax.random.normal(k, (num_samples, 1))
            
            # Physical: loc + scale * x_std
            x_phys = dist.loc + dist.scale * x_std
            
        else:
            raise ValueError(f"Unsupported distribution: {type(dist)}")
            
        X_std_list.append(x_std)
        X_phys_list.append(x_phys)
        
    X_std = jnp.hstack(X_std_list) # (N, dims)
    # X_phys maps index i to the i-th uncertain parameter
    
    # 3. Evaluate Physical Model
    # We need to construct the full parameter array for each sample
    
    # Helper to assemble params array
    def assemble_params(uncertain_vals_phys):
        # uncertain_vals_phys: (dims,)
        full_params = []
        uncertain_idx = 0
        
        for name in parameter_names:
            if name in uncertain_params:
                full_params.append(uncertain_vals_phys[uncertain_idx])
                uncertain_idx += 1
            else:
                full_params.append(fixed_params[name])
                
        return jnp.concatenate([jnp.atleast_1d(p) for p in full_params])

    # Vmap assembly
    # Transpose list of arrays is annoying, assume scalar uncertain params for now? 
    # Yes, Distribution classes defined scalar ranges mostly.
    
    # We iterate over samples
    X_phys_matrix = jnp.hstack(X_phys_list) # (N, dims)
    
    params_batch = jax.vmap(assemble_params)(X_phys_matrix)
    
    # Evaluate model
    # model_func(params, acq)
    # Vmap over params
    signals = jax.vmap(lambda p: model_func(p, acquisition))(params_batch)
    # signals shape: (N_samples, N_measurements)
    
    # 4. Fit Surrogate
    # We fit the surrogate mapping X_std -> Signal
    # Note: signals is vector output. PCE supports scalar output.
    # We can fit independent PCEs for each measurement channel?
    # Or just vectorize the coefficients?
    
    # Implementation of PCE.fit assumes `values` is (N,), scalar output.
    # We need to adapt `PolynomialChaosExpansion` to support vector outputs or loop here.
    # Vector output support is better.
    # Let's verify `surrogate.py`: coefficients are `Float[Array, " num_terms"]`.
    # If we want vector output `coefficients` should be `Float[Array, " num_terms num_signals"]`.
    
    # Modifying surrogate.py is one option, or vmapping the fit classmethod.
    # vmapping the fit classmethod over output dimensions (N_measurements).
    # fit(parameters, values)
    # values: (N_samples, N_meas) -- transpose to (N_meas, N_samples)
    
    signals_T = signals.T # (N_meas, N_samples)
    
    # We want to fit surrogate for each measurement.
    # X_std is shared.
    
    # Let's just create a loop or vmap. 
    # Use vmap on the fit method? fit is classmethod.
    
    # Or simple: Compute mean/variance directly from samples if we only care about stats?
    # No, gPC is more accurate/convergent for smooth models than Monte Carlo.
    # BUT fitting gPC then extracting stats is equivalent to MC if we use random samples for fitting?
    # Actually, if we use regression, we get the coefficients.
    # Mean = c_0. Variance = sum c_k^2 * <Phi_k^2>.
    
    # Since we implemented Least Squares fitting, we can run it.
    
    # If we have many measurements (e.g. 100), fitting 100 surrogates is fast.
    
    from functools import partial
    
    # fit signature: fit(parameters, values, distributions, total_order)
    # values: (N_samples,)
    
    fit_fn = lambda y: PolynomialChaosExpansion.fit(X_std, y, dist_names, order)
    surrogates = list(map(fit_fn, signals_T)) # Can't vmap classes easily returning PyTrees of Modules yet? Equinox supports it.
    
    # 5. Extract Statistics
    means = []
    variances = []
    
    for surr in surrogates:
        coeffs = surr.coefficients
        indices = surr.basis_indices
        
        # Mean is the coefficient of the zeroth order term (all indices 0)
        # Assuming index 0 is always the constant term? 
        # create_basis_indices usually puts constant first, but we must verify.
        # Let's find index where all are 0.
        
        const_mask = jnp.all(indices == 0, axis=1)
        mean_val = jnp.sum(coeffs * const_mask) # Should be just one term
        
        # Variance
        # Orthogonality: E[Phi_i Phi_j] = delta_ij * E[Phi_i^2]
        # Var[f] = sum_{k != 0} c_k^2 E[Phi_k^2]
        
        # We need the norms of the basis polynomials.
        # Legendre Pn: \int_{-1}^1 P_n^2 dx / 2 = 1/(2n+1). 
        # Wait, P_n are orthogonal w.r.t Uniform[-1, 1], so weight is 1/2.
        # Norm^2 = 1/(2n+1).
        
        # Hermite He_n: \int He_n^2 e^-x^2/2 dx / sqrt(2pi) = n!
        # Norm^2 = n!
        
        # Basis term Phi_k is product of univariate polys.
        # Norm_k^2 = prod_d Norm_{k,d}^2
        
        norms_sq = jnp.ones(indices.shape[0])
        
        for dim_idx, dist_name in enumerate(dist_names):
            degs = indices[:, dim_idx] # (num_terms,)
            
            if dist_name == 'Uniform':
                # Legendre norms
                # N_n = 1 / (2n+1)
                term_norms = 1.0 / (2.0 * degs + 1.0)
            elif dist_name == 'Normal':
                # Hermite norms (Probabilists)
                # N_n = n!
                term_norms = jnp.exp(jax.scipy.special.gammaln(degs + 1))
            
            norms_sq = norms_sq * term_norms
            
        # Variance calculation
        # sum c_k^2 * norms_sq (excluding mean)
        var_val = jnp.sum((coeffs ** 2 * norms_sq) * (1 - const_mask))
        
        means.append(mean_val)
        variances.append(var_val)
        
    return jnp.array(means), jnp.sqrt(jnp.array(variances))
