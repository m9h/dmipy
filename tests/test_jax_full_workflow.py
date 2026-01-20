
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def test_full_workflow():
    # 1. Define Models
    stick = Stick()
    ball = G1Ball()
    
    # 2. Compose MCM
    model = JaxMultiCompartmentModel([stick, ball])
    
    # 3. Define Acquisition
    # Create simple shell
    # USE SI UNITS: s/m^2. 1000 s/mm^2 = 1e9 s/m^2
    bvals = jnp.array([0.0] + [1e9] * 6 + [2e9] * 6)
    # Simple gradients
    # [1,0,0], [0,1,0], [0,0,1] etc.
    # Just randoms for test
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (13, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs = bvecs.at[0].set(jnp.array([0., 0., 0.])) # b0
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # 4. Synthesize Data
    # True Params
    # Stick: mu=[0.5, 0.5] (rad), lambda_par=1.7e-9
    # Ball: lambda_iso=3.0e-9
    # Fractions: stick=0.6, ball=0.4 (implied)
    
    true_params = {
        'mu': jnp.array([0.5, 0.5]),
        'lambda_par': 1.7e-9,
        'lambda_iso': 3.0e-9,  # Note: collision handling in MCM might rename this if collision exists
                               # Here 'lambda_iso' vs 'lambda_par', no collision.
        'partial_volume_0': 0.6,
        'partial_volume_1': 0.4
    }
    
    # Convert to array
    # We rely on the model's ordering
    # But wait, we can't easily construct the array manually without knowing order.
    # Actually modeling_framework `parameter_dictionary_to_array` handles this!
    
    true_params_array = model.parameter_dictionary_to_array(true_params)
    
    # Generate signal
    data = model.model_func(true_params_array, acq)
    
    # Add noise? No, let's test noiseless recovery first to verify optimization topology handling.
    
    # 5. Fit
    print("Fitting...")
    fitted_dict = model.fit(acq, data)
    
    print("True Params:", true_params)
    print("Fitted Params:", fitted_dict)
    
    # 6. Verify
    # Check orientation (mu) - handle periodicity/antipodal? 
    # Stick is antipodal symmetric. [0.5, 0.5] vs others.
    # Diffusivities should match.
    
    tol = 1e-1 # Fairly loose for crude random init+optimization?
               # Should be tight for noiseless if init works.
    
    # lambda_par
    err_d_par = jnp.abs(fitted_dict['lambda_par'] - true_params['lambda_par'])
    print(f"Error lambda_par: {err_d_par}")
    assert err_d_par < 1e-10, "Failed to recover lambda_par"

    # lambda_iso
    err_d_iso = jnp.abs(fitted_dict['lambda_iso'] - true_params['lambda_iso'])
    print(f"Error lambda_iso: {err_d_iso}")
    assert err_d_iso < 1e-10, "Failed to recover lambda_iso"
    
    # Fractions
    err_f0 = jnp.abs(fitted_dict['partial_volume_0'] - true_params['partial_volume_0'])
    print(f"Error f0: {err_f0}")
    assert err_f0 < 1e-2, "Failed to recover fraction"

if __name__ == "__main__":
    test_full_workflow()
    print("Test Passed!")
