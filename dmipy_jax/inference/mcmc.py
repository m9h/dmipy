import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import blackjax
from functools import partial

def rician_log_likelihood(params, model_func, acquisition, data, sigma):
    """
    Computes the Rician Log-Likelihood of the data given the model parameters.

    Args:
        params (jax.numpy.ndarray): Model parameters (1D array).
        model_func (callable): Function that takes (params, acquisition) and returns predicted signal.
        acquisition (JaxAcquisition): Acquisition scheme object.
        data (jax.numpy.ndarray): Observed data (1D array).
        sigma (float): Noise standard deviation (scalar or broadcastable).

    Returns:
        float: Sum of log-likelihoods over all measurements.
    """
    S_model = model_func(params, acquisition)
    
    # Rician Log-Likelihood
    # ln(L) = ln(data / sigma^2) - (data^2 + S_model^2) / (2 * sigma^2) + ln(I0(data * S_model / sigma^2))
    
    # Improve numerical stability for ln(I0(z)) using i0e:
    # I0(z) = i0e(z) * exp(|z|)
    # ln(I0(z)) = ln(i0e(z)) + |z|
    
    z = (data * S_model) / (sigma**2)
    ln_I0_z = jnp.log(jsp.i0e(z)) + jnp.abs(z)
    
    # We can omit constant terms that don't depend on parameters for optimization/sampling,
    # but for true log-likelihood calculation, we keep them.
    
    ll = jnp.log(data / (sigma**2)) - (data**2 + S_model**2) / (2 * sigma**2) + ln_I0_z
    
    return jnp.sum(ll)


def fit_mcmc(model_func, data, acquisition, initial_params, sigma, n_samples=1000, n_warmup=500, rng_key=None):
    """
    Fits a JAX model to data using MCMC (NUTS sampler from Blackjax).

    Args:
        model_func (callable): Function to fit.
        data (jax.numpy.ndarray): 1D or (N, M) array of data. If (N, M), vmap is used over N voxels.
        acquisition (JaxAcquisition): Acquisition scheme.
        initial_params (jax.numpy.ndarray): Initial guess for parameters.
            If data is (N, M), this should be (N, P) or broadcastable.
        sigma (float): Noise level.
        n_samples (int): Number of posterior samples to draw.
        n_warmup (int): Number of warmup steps.
        rng_key (jax.random.PRNGKey): JAX PRNG key. Checks global if None (not recommended for strict reproducibility).

    Returns:
        dict: Posterior samples for each parameter.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Determine if we are fitting a single voxel or multiple
    is_batched = data.ndim > 1
    
    if not is_batched:
        # Reshape to batch size 1 for uniform handling, or just run single
        # But user requirements ask for vmap support.
        # Let's implement single voxel logic first and then vmap it.
        pass

    # Core Sampling Logic for a single voxel
    def _run_chain(rng_key, y, init_p, sig):
        
        def logdensity_fn(params):
            return rician_log_likelihood(params, model_func, acquisition, y, sig)
        
        # Initialize NUTS
        # We need an inverse mass matrix. A diagonal one is a good default.
        inverse_mass_matrix = jnp.ones(init_p.shape)
        step_size = 1e-3 # Initial step size, simplified adaptation might be needed
        
        # Blackjax NUTS interface
        # We'll use the window adaptation which handles step size and mass matrix
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, parameters), _ = warmup.run(rng_key, init_p, n_warmup)
        
        # Sampling kernel
        kernel = blackjax.nuts(logdensity_fn, **parameters).step
        
        # Sampling loop using scan
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state.position
            
        keys = jax.random.split(rng_key, n_samples)
        _, samples = jax.lax.scan(one_step, state, keys)
        
        return samples

    if is_batched:
        # Vmap the chain runner
        # y: (N, M), init_p: (N, P), sig: scalar or (N, 1) or (N,)
        
        # Handle sigma broadcasting if scalar
        if jnp.ndim(sigma) == 0:
            batch_size = data.shape[0]
            sigma_in = jnp.full((batch_size,), sigma)
        else:
            sigma_in = sigma
            
        # Split keys for each voxel
        batch_size = data.shape[0]
        keys = jax.random.split(rng_key, batch_size)
        
        return jax.vmap(_run_chain)(keys, data, initial_params, sigma_in)
    else:
        return _run_chain(rng_key, data, initial_params, sigma)
