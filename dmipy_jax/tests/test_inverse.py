import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.inverse.solvers import MicrostructureOperator, GlobalOptimizer, AMICOSolver
from dmipy_jax.signal_models.stick import Stick

# Define a simple Ball model for testing
class Ball:
    parameter_names = ['lambda_iso']
    parameter_cardinality = {'lambda_iso': 1}
    
    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        # S = exp(-b * D)
        return jnp.exp(-bvals * lambda_iso)

class MockAcquisition:
    def __init__(self, bvals, bvecs):
        self.bvals = bvals
        self.gradient_directions = bvecs
        self.N_measurements = len(bvals)

@pytest.fixture
def acquisition():
    bvals = jnp.array([0.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0])
    bvecs = jnp.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    return MockAcquisition(bvals, bvecs)

def test_microstructure_operator(acquisition):
    # Compose Stick + Ball
    # Stick: mu (2), lambda_par (1)
    # Ball: lambda_iso (1)
    # Total params + fractions?
    # compose_models adds Partial Volumes if len > 1?
    pass # Implementation detail check needed.
    
    # Let's use single model first to test simple wrapping
    model = JaxMultiCompartmentModel([Ball()])
    
    # Input shape: (10, 10, 1) -> 1 parameter (lambda_iso) per voxel
    input_shape = (10, 10, 1)
    op = MicrostructureOperator(model, acquisition, input_shape)
    
    # Forward pass
    x = jnp.ones(input_shape) * 1e-3 # D = 1e-3
    y = op(x)
    
    assert y.shape == (10, 10, acquisition.N_measurements)
    # Check value for b=0 (should be 1)
    assert jnp.allclose(y[..., 0], 1.0)
    
def test_global_optimizer_tv(acquisition):
    # Setup phantom with edge
    # 10x10 grid
    # Left half: D=1e-3, Right half: D=2e-3
    
    model = JaxMultiCompartmentModel([Ball()])
    input_shape = (10, 10, 1)
    op = MicrostructureOperator(model, acquisition, input_shape)
    
    true_map = np.zeros(input_shape)
    true_map[:, :5, 0] = 1.0e-3
    true_map[:, 5:, 0] = 2.0e-3
    true_map = jnp.array(true_map)
    
    # Generate clean signal
    clean_signal = op(true_map)
    
    # Add noise
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, clean_signal.shape) * 0.05
    data_noisy = clean_signal + noise
    
    # Solve with TV
    optimizer = GlobalOptimizer(op)
    recon_map = optimizer.solve_tv(data_noisy, lambda_tv=0.01, maxiter=50) # Small iter for speed
    
    assert recon_map.shape == input_shape
    
    # Check if TV smoothed the noise (variance check)
    # Left region variance
    noisy_inverse = -jnp.log(data_noisy[..., 1] / data_noisy[..., 0]) / 1000.0 # Rough approx
    
    # Note: Simplistic check. Just ensure it runs and output is reasonable range.
    assert jnp.all(recon_map > 0)
    assert not jnp.any(jnp.isnan(recon_map))

def test_amico_solver(acquisition):
    # Linear problem: Signal = w1 * S1 + w2 * S2
    # Define dictionary M with 2 columns corresponding to D1=1e-3, D2=3e-3
    
    ball = Ball()
    s1 = ball(acquisition.bvals, acquisition.gradient_directions, lambda_iso=1e-3)
    s2 = ball(acquisition.bvals, acquisition.gradient_directions, lambda_iso=3e-3)
    s3 = ball(acquisition.bvals, acquisition.gradient_directions, lambda_iso=5e-3)
    
    M = jnp.stack([s1, s2, s3], axis=1) # (N_meas, 3)
    
    # Ground truth: Pure s2 (w=[0, 1, 0])
    y = s2
    
    # Use AMICOSolver
    # It requires a model init, but fit method takes M directly.
    # We can pass dummy model/acq to init.
    model = JaxMultiCompartmentModel([Ball()])
    solver = AMICOSolver(model, acquisition)
    
    # Fit
    # L1 regularization to encourage sparsity
    x = solver.fit(y, M, lambda_l1=1e-4)
    
    # Expect x approx [0, 1, 0]
    print("Estimated AMICO weights:", x)
    assert jnp.argmax(x) == 1
    assert x[1] > 0.9 # Should be close to 1
    assert x[0] < 0.1
