
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.signal_models.sandi import SphereGPD, get_sandi_model
from dmipy_jax.acquisition import JaxAcquisition

# Mock data
@pytest.fixture
def mock_acquisition():
    # 2 shells: b=1000, b=2500
    bvals = jnp.array([0.0, 1000.0, 1000.0, 2500.0, 2500.0]) * 1e6 # s/m^2
    bvecs = jnp.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    # Delta/delta
    delta = 0.010 # 10 ms
    Delta = 0.030 # 30 ms
    
    return JaxAcquisition(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=delta, 
        Delta=Delta
    )

def test_sphere_gpd_basic_properties(mock_acquisition):
    # Test initialization
    sphere = SphereGPD(diameter=10e-6, diffusion_constant=3e-9)
    
    # Test call
    signal = sphere(
        bvals=mock_acquisition.bvalues,
        gradient_directions=mock_acquisition.gradient_directions,
        acquisition=mock_acquisition
    )
    
    # Validations
    assert signal.shape == mock_acquisition.bvalues.shape
    assert jnp.all(signal >= 0.0)
    assert jnp.all(signal <= 1.0)
    assert signal[0] == 1.0 # b=0 should be 1
    
    # Check monotonicity/attenuation?
    # For b=2500, signal should be lower/equal to b=1000 for same direction
    # (assuming GPD behaves monotonically with b for fixed delta/Delta)
    assert signal[3] <= signal[1] + 1e-6

def test_sandi_model_shape_and_range(mock_acquisition):
    sandi_model = get_sandi_model()
    
    # Params: [theta, phi, f_stick, f_sphere, f_ball, diameter, lambda_perp]
    # Fractions sum < 1 (f_zeppelin takes remainder)
    params = jnp.array([
        0.0, 0.0, # mu along x
        0.3, 0.2, 0.1, # f_stick, f_sphere, f_ball -> f_zeppelin = 1 - 0.6 = 0.4
        8e-6, # diameter
        0.5e-9 # lambda_perp
    ])
    
    signal = sandi_model(params, mock_acquisition)
    
    assert signal.shape == mock_acquisition.bvalues.shape
    assert jnp.all(signal >= 0.0)
    assert jnp.all(signal <= 1.0 + 1e-6) # tolerance
    
def test_sandi_gradients(mock_acquisition):
    sandi_model = get_sandi_model()
    
    params = jnp.array([0.0, 0.0, 0.3, 0.2, 0.1, 8e-6, 0.5e-9])
    
    def loss(p):
        return jnp.sum(sandi_model(p, mock_acquisition))
        
    grad_fn = jax.grad(loss)
    grads = grad_fn(params)
    
    assert jnp.all(jnp.isfinite(grads))
    print("Gradients:", grads)

if __name__ == "__main__":
    # Manual run for quick feedback
    # Create mock acq
    acq = mock_acquisition()
    test_sphere_gpd_basic_properties(acq)
    test_sandi_model_shape_and_range(acq)
    test_sandi_gradients(acq)
    print("All generic tests passed!")
