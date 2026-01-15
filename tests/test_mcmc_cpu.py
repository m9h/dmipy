import os
# Force JAX to use CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from dmipy_jax.inference.mcmc import fit_mcmc


def test_mcmc_cpu_simple():
    """
    Fast integration test for Rician MCMC on CPU.
    """
    # 1. Setup Mock Data and Model
    # Simple linear model: S = m * x + c
    # We'll just estimate one parameter 'm' for simplicity, assuming x=1, c=0
    # Actually, simpler: Constant model S = theta
    
    true_theta = 0.5
    N_measurements = 20
    sigma = 0.1
    
    # Mock Acquisition (just a dummy object, as our model won't use it complexly)
    class MockAcquisition:
        pass
    acquisition = MockAcquisition()
    
    # Model Function: returns constant signal equal to param[0]
    # params is shape (1,)
    def model_func(params, acquisition):
        # Broadcast params[0] to match data size (conceptually)
        # But wait, rician_log_likelihood calculates S_model = model_func(...)
        # and then does (data * S_model). So S_model should be same shape as data.
        return jnp.full((N_measurements,), params[0])

    # Generate Synthetic Data with Rician Noise
    # clean signal
    S_clean = np.full((N_measurements,), true_theta)
    
    # Rician noise: sqrt((S + n1)^2 + n2^2), where n1, n2 ~ N(0, sigma)
    rng = np.random.default_rng(42)
    n1 = rng.normal(0, sigma, size=N_measurements)
    n2 = rng.normal(0, sigma, size=N_measurements)
    data = np.sqrt((S_clean + n1)**2 + n2**2)
    
    # Convert to JAX array
    data_jax = jnp.array(data)
    
    # Initial Params
    initial_params = jnp.array([0.1]) # Start far from 0.5
    
    # 2. Run MCMC (Nano Chain)
    print("\nRunning MCMC on CPU...")
    # fit_mcmc returns samples. 
    # Based on fit_mcmc code: returns samples (dict or array? Wrapper said "Posterior samples for each parameter")
    # Actually, looking at the code I read:
    # return samples
    # samples is from blackjax...scan. usually it matches the structure of 'initial_params' if that was the state.
    # The 'state' in blackjax nuts is complex, but 'samples' returned from the scan loop 
    # in `_run_chain` is `state.position`.
    # So it should be a JAX array of shape (n_samples, n_params)
    
    samples = fit_mcmc(
        model_func=model_func,
        data=data_jax,
        acquisition=acquisition,
        initial_params=initial_params,
        sigma=sigma,
        n_samples=10,
        n_warmup=10,
        rng_key=jax.random.PRNGKey(123)
    )
    
    # 3. Assertions
    
    # Assert output shape
    # fit_mcmc returns `samples`. Since initial_params was (1,), samples should be (10, 1)
    assert samples.shape == (10, 1), f"Expected shape (10, 1), got {samples.shape}"
    
    # Assert samples are not all identical (it moved)
    # varying over the chain
    std_samples = jnp.std(samples)
    assert std_samples > 0.0, "MCMC chain did not move (all samples identical)."
    
    # Assert no NaNs
    assert not jnp.any(jnp.isnan(samples)), "NaNs found in MCMC chain."
    
    print("MCMC CPU Test Passed: Chain successfully explored parameter space.")

if __name__ == "__main__":
    test_mcmc_cpu_simple()
