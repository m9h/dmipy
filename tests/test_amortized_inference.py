
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from dmipy_jax.inference.amortized import ZeppelinNetwork, self_supervised_loss
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.zeppelin import Zeppelin

def test_zeppelin_network_output_structure():
    """Test that the network produces the correct output structure and shapes."""
    key = jax.random.PRNGKey(0)
    n_measurements = 10
    network = ZeppelinNetwork(key, n_measurements)
    
    # Random signal input
    signal = jax.random.normal(key, (n_measurements,))
    
    # Forward pass
    params = network(signal)
    
    # Check keys
    expected_keys = {'lambda_par', 'lambda_perp', 'fraction', 'mu'}
    assert set(params.keys()) == expected_keys
    
    # Check shapes and constraints
    assert params['lambda_par'].ndim == 0
    assert params['lambda_perp'].ndim == 0
    assert params['fraction'].ndim == 0
    assert params['mu'].shape == (2,)
    
    # Constraints check
    assert params['lambda_par'] > 0
    assert params['lambda_perp'] > 0
    assert 0 <= params['fraction'] <= 1

def test_self_supervised_loss_computation():
    """Test that the loss function runs and returns a scalar."""
    key = jax.random.PRNGKey(1)
    n_measurements = 15
    
    # Setup acquisition
    bvals = jnp.linspace(0, 3000, n_measurements)
    # Random gradient directions (normalized)
    vecs = jax.random.normal(key, (n_measurements, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=vecs)
    
    # Setup Network
    network = ZeppelinNetwork(key, n_measurements)
    
    # Synthesize data using Zeppelin model with known params
    gt_lambda_par = 1.7e-9
    gt_lambda_perp = 0.2e-9
    gt_mu = jnp.array([jnp.pi/2, 0.0]) # x-axis
    
    zeppelin = Zeppelin(mu=gt_mu, lambda_par=gt_lambda_par, lambda_perp=gt_lambda_perp)
    data = zeppelin(bvals=bvals, gradient_directions=vecs)
    
    # Verify synthetic data is valid (no NaNs)
    assert not jnp.any(jnp.isnan(data))
    
    # Calculate Loss
    loss = self_supervised_loss(network, data, acq)
    
    # Check loss is a scalar float
    assert loss.ndim == 0
    assert not jnp.isnan(loss)
    
def test_jit_compatibility():
    """Test that the loss function can be JIT compiled."""
    key = jax.random.PRNGKey(2)
    n_measurements = 10
    
    # Minimal Setup
    bvals = jnp.linspace(0, 1000, n_measurements)
    vecs = jnp.ones((n_measurements, 3)) / jnp.sqrt(3)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=vecs)
    network = ZeppelinNetwork(key, n_measurements)
    data = jnp.ones(n_measurements)
    
    # JIT compile the loss using eqx.filter_jit (required for modules with static fields like functions)
    jit_loss = eqx.filter_jit(self_supervised_loss)
    
    # Run jitted function
    val = jit_loss(network, data, acq)
    assert val.ndim == 0

if __name__ == "__main__":
    # If run as script, execute tests
    test_zeppelin_network_output_structure()
    test_self_supervised_loss_computation()
    test_jit_compatibility()
    print("All tests passed!")
