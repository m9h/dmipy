import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import blackjax
from functools import partial
import equinox as eqx
from typing import Callable, Optional, Union, Dict

class MCMCInference(eqx.Module):
    """
    MCMC Inference using Blackjax (NUTS).
    """
    model_func: Callable
    acquisition: object
    sigma: float
    n_samples: int = 1000
    n_warmup: int = 500

    def __init__(
        self, 
        model_func: Callable, 
        acquisition: object, 
        sigma: float = 0.02, # Default approximate noise level
        n_samples: int = 1000,
        n_warmup: int = 500
    ):
        self.model_func = model_func
        self.acquisition = acquisition
        self.sigma = sigma
        self.n_samples = n_samples
        self.n_warmup = n_warmup

    def fit(
        self, 
        data: jnp.ndarray, 
        initial_params: jnp.ndarray, 
        rng_key: Optional[jax.random.PRNGKey] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Fits the model to the data using NUTS MCMC.

        Args:
            data (jax.numpy.ndarray): Observed data (N_voxels, N_measurements) or (N_measurements,).
            initial_params (jax.numpy.ndarray): Initial guess (N_voxels, N_params) or (N_params,).
            rng_key (jax.random.PRNGKey): Random key.

        Returns:
            dict: Dictionary containing 'samples' (posterior samples).
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        is_batched = data.ndim > 1
        
        if is_batched:
            # Batched execution (vmap)
            batch_size = data.shape[0]
            keys = jax.random.split(rng_key, batch_size)
            
            # Handle sigma if it's a scalar, broadcast it
            sigma_in = self.sigma
            if jnp.ndim(sigma_in) == 0:
                 sigma_in = jnp.full((batch_size,), sigma_in)

            samples = jax.vmap(self._run_chain)(keys, data, initial_params, sigma_in)
            return {"samples": samples}
        else:
            # Single voxel execution
            samples = self._run_chain(rng_key, data, initial_params, self.sigma)
            return {"samples": samples}

    def _run_chain(self, rng_key, y, init_p, sig):
        """
        Runs a single MCMC chain for one voxel.
        """
        def logdensity_fn(params):
            return rician_log_likelihood(params, self.model_func, self.acquisition, y, sig)
        
        # Initialize NUTS
        # Inverse mass matrix: typically diagonal. 
        # We start with ones. Adaptation will update it.
        inverse_mass_matrix = jnp.ones(init_p.shape)
        
        # Window adaptation
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        
        # Run warmup
        (state, parameters), _ = warmup.run(rng_key, init_p, self.n_warmup)
        
        # Build kernel with adapted parameters
        kernel = blackjax.nuts(logdensity_fn, **parameters).step
        
        # Sampling loop
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, state.position
            
        keys = jax.random.split(rng_key, self.n_samples)
        _, samples = jax.lax.scan(one_step, state, keys)
        
        return samples

def rician_log_likelihood(params, model_func, acquisition, data, sigma):
    """
    Computes the Rician Log-Likelihood.
    """
    S_model = model_func(params, acquisition)
    
    # Rician LL: ln(L) = ln(d/s^2) - (d^2 + S^2)/(2s^2) + ln(I0(dS/s^2))
    # using I0(z) ~ i0e(z) * exp(|z|) to avoid overflow
    
    z = (data * S_model) / (sigma**2)
    ln_I0_z = jnp.log(jsp.i0e(z)) + jnp.abs(z)
    
    # We ignore constant terms for MCMC sampling efficiency if possible, 
    # but exact LL is useful for comparison.
    # Term 1: ln(data/sigma^2) -> constant w.r.t params
    # Term 2: -(data^2 + S_model^2) / (2 * sigma^2)
    # Term 3: ln_I0_z
    
    # log_likelihood = term1 + term2 + term3
    
    # Implementation:
    ll = -(data**2 + S_model**2) / (2 * sigma**2) + ln_I0_z
    
    # Note: excluding the strictly constant term ln(data/sigma^2) 
    # does not change the shape of the posterior for parameter sampling.
    
    return jnp.sum(ll)
