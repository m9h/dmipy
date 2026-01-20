import jax
import jax.numpy as jnp
import pytest
import numpy as np
from dmipy_jax.signal_models import TortuosityModel, RestrictedCylinder
from dmipy_jax.core import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def test_tortuosity_model():
    """Test instantiation and basic signal prediction of TortuosityModel."""
    model = TortuosityModel()
    assert 'mu' in model.parameter_names
    assert 'lambda_par' in model.parameter_names
    assert 'icvf' in model.parameter_names
    
    # Create simple acquisition
    bvals = jnp.array([0.0, 1000.0, 2000.0])
    bvecs = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    acq = JaxAcquisition(bvals, bvecs)
    
    # Call directly
    # mu along Z: theta=0, phi=0
    mu = jnp.array([0.0, 0.0])
    lambda_par = 1.7e-9
    icvf = 0.5
    
    signal = model(bvals=bvals, gradient_directions=bvecs, mu=mu, lambda_par=lambda_par, icvf=icvf)
    assert signal.shape == (3,)
    assert jnp.all(signal >= 0.0)
    assert jnp.all(signal <= 1.0)
    
    # Check physics: b=0 -> 1
    assert jnp.isclose(signal[0], 1.0)

def test_restricted_cylinder():
    """Test RestrictedCylinder wrapper."""
    model = RestrictedCylinder()
    
    bvals = jnp.array([0.0, 3000.0])
    bvecs = jnp.array([[1,0,0], [0,0,1]])
    # Needs Delta/delta
    acq = JaxAcquisition(bvals, bvecs, big_delta=0.04, small_delta=0.03)
    
    mu = jnp.array([0.0, 0.0])
    lambda_par = 1.7e-9
    diameter = 5e-6
    
    signal = model(bvals=bvals, gradient_directions=bvecs, 
                   mu=mu, lambda_par=lambda_par, diameter=diameter,
                   big_delta=0.04, small_delta=0.03)
                   
    assert signal.shape == (2,)

def test_global_initialization_fit():
    """Test GlobalBruteInitializer via JaxMultiCompartmentModel.fit()."""
    
    # 1. Setup Model: Stick + Ball
    # Note: Tortuosity is a bit complex for simple fit test, use Stick+Ball or Stick+Zeppelin
    # But let's try TortuosityModel to verification it works in MCM
    
    model = JaxMultiCompartmentModel([TortuosityModel()])
    # TortuosityModel params: mu, lambda_par, icvf + partial_volume_0
    
    # 2. Synthetic Data
    # True params
    mu_true = jnp.array([1.57, 0.0]) # pi/2, 0 -> X axis
    d_par_true = 2.0e-9
    icvf_true = 0.7
    f_true = 1.0 # only one model
    
    # Create dictionary
    params_true = {
        'mu': mu_true,
        'lambda_par': d_par_true,
        'icvf': icvf_true,
        'partial_volume_0': f_true
    }
    
    bvals = jnp.array([0, 1000, 1000, 1000, 3000, 3000])
    # directions: aligned X, aligned Y, aligned Z
    bvecs = jnp.array([
        [0,0,1],
        [1,0,0], [0,1,0], [0,0,1],
        [1,0,0], [0,1,0]
    ])
    acq = JaxAcquisition(bvals, bvecs)
    
    # Generate signal
    flat_params = model.parameter_dictionary_to_array(params_true)
    data = model.model_func(flat_params, acq)
    
    # 3. Fit
    # Fit should use GlobalBrute to find approx solution then optimize
    fitted_dict = model.fit(acq, data, compute_uncertainty=False)
    
    # 4. Check results
    # Orientation ambiguity?
    # lambda_par should be close
    print("True:", params_true)
    print("Fitted:", fitted_dict)
    
    assert jnp.isclose(fitted_dict['lambda_par'], d_par_true, rtol=0.2)
    assert jnp.isclose(fitted_dict['icvf'], icvf_true, rtol=0.2)

if __name__ == "__main__":
    test_tortuosity_model()
    test_restricted_cylinder()
    test_global_initialization_fit()
    print("All tests passed!")
