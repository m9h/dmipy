import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.inference.mcmc import MCMCInference
from dmipy_jax.acquisition import JaxAcquisition

def simple_model(params, acquisition):
    # params: [signal_value]
    # Simple model: Returns constant signal equal to params[0]
    # In a real scenario, this would depend on acquisition
    return jnp.full((acquisition.N,), params[0])

def test_mcmc_simple_recovery():
    """
    Test if MCMC can recover the mean of a simple constant signal with Rician noise.
    """
    # Mock Acquisition
    class MockAcq:
        N = 100
    acquisition = MockAcq()

    # Ground Truth
    true_signal_val = 1.0
    sigma = 0.1
    
    # Generate Synthetic Data
    # For high SNR, Rician ~ Gaussian
    rng = jax.random.PRNGKey(42)
    noise = jax.random.normal(rng, (acquisition.N,)) * sigma
    data = jnp.full((acquisition.N,), true_signal_val) + noise
    # Ensure positive for Rician
    data = jnp.abs(data)

    # Initialize MCMC
    mcmc = MCMCInference(
        model_func=simple_model,
        acquisition=acquisition,
        sigma=sigma,
        n_samples=500,
        n_warmup=200
    )

    # Initial guess
    initial_params = jnp.array([0.5]) # Start far from 1.0

    # Fit
    rng_fit = jax.random.PRNGKey(101)
    results = mcmc.fit(data, initial_params, rng_key=rng_fit)
    samples = results['samples'] # Shape (n_samples, 1) or (n_samples, )

    # Check shape
    assert samples.shape[0] == 500

    # Check convergence (mean of posterior should be close to true value)
    posterior_mean = jnp.mean(samples)
    print(f"Posterior Mean: {posterior_mean}, True: {true_signal_val}")
    
    # 0.1 sigma means standard error of mean is very small for 100 points
    assert jnp.abs(posterior_mean - true_signal_val) < 0.05

def test_mcmc_batched_recovery():
    """
    Test batched execution (vmap).
    """
    class MockAcq:
        N = 50
    acquisition = MockAcq()
    
    # Two voxels: Voxel 0 -> 1.0, Voxel 1 -> 2.0
    true_vals = jnp.array([[1.0], [2.0]])
    sigma = 0.1
    
    rng = jax.random.PRNGKey(42)
    noise = jax.random.normal(rng, (2, acquisition.N)) * sigma
    
    # Broadcasting true_vals correctly: (2, 1) -> (2, 50)
    data = true_vals + noise
    data = jnp.abs(data)
    
    mcmc = MCMCInference(
        model_func=simple_model,
        acquisition=acquisition,
        sigma=sigma,
        n_samples=200,
        n_warmup=100
    )
    
    init_params = jnp.array([[0.5], [2.5]])
    
    results = mcmc.fit(data, init_params, rng_key=jax.random.PRNGKey(99))
    samples = results['samples'] # Shape (2, n_samples, 1)
    
    assert samples.shape == (2, 200, 1)
    
    means = jnp.mean(samples, axis=1)
    assert jnp.allclose(means, true_vals, atol=0.1)
