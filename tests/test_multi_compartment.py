import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def test_multi_compartment_fit():
    # 1. Setup Acquisition
    # Create simple acquisition
    # b=0, then some shells
    bvalues = jnp.array([0., 1000., 1000., 1000., 2000., 2000., 2000.])
    # Random-ish directions
    bvecs = jnp.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0.707, 0.707, 0.],
        [0.707, 0., 0.707],
        [0., 0.707, 0.707]
    ])
    
    # Normalize
    norms = jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    # Avoid divide by zero
    norms = jnp.where(norms == 0, 1.0, norms)
    bvecs = bvecs / norms
    
    # Clean up b=0 case (norm was 0, bvec becomes 0/1=0)
    # Actually if b=0, bvec doesn't matter much but dmipy expects normalized or zero
    
    acq = JaxAcquisition(bvalues=bvalues, gradient_directions=bvecs)
    
    # 2. Setup Models
    stick = Stick()
    ball = G1Ball()
    
    # Manually attach ranges for testing since defaults might be missing in those classes
    # 0.6 is roughly 3e-9 / 5e-9 scale?
    # Typical diffusivity: 2e-9 m^2/s = 0.002 mm^2/s.
    # b-values are usually s/mm^2 or s/m^2.
    # If b=1000 s/mm^2, D=2e-3 mm^2/s -> bD = 2.
    # If b=1e9 s/m^2, D=2e-9 m^2/s -> bD = 2.
    # dmipy usually uses SI units (seconds, meters).
    # So D ~ 1e-9.
    
    stick.parameter_ranges = {
        'mu': [(0.0, jnp.pi), (-jnp.pi, jnp.pi)], 
        'lambda_par': (0.1e-9, 3.0e-9)
    }
    ball.parameter_ranges = {
        'lambda_iso': (0.1e-9, 3.0e-9)
    }
    
    # 3. Combine
    mcm = JaxMultiCompartmentModel([stick, ball])
    
    print("Parameter names:", mcm.parameter_names)
    
    # 4. Generate Synthetic Data
    true_params = {
        'mu': jnp.array([1.57, 0.0]), # ~X-axis
        'lambda_par': jnp.array(2.0e-9),
        'lambda_iso': jnp.array(1.0e-9),
        'partial_volume_0': jnp.array(0.6),
        'partial_volume_1': jnp.array(0.4)
    }
    
    # Convert manually to check ordering if needed, but rely on helper
    params_array = mcm.parameter_dictionary_to_array(true_params)
    signal = mcm.model_func(params_array, acq)
    
    # 5. Fit
    # Use noiseless data for verification of optimizer correctness
    fitted_params = mcm.fit(acq, signal)
    
    # 6. Verify
    print("True params:", true_params)
    print("Fitted params:", fitted_params)
    
    # Tolerances
    rtol = 0.05
    atol_diff = 1e-10
    
    assert jnp.allclose(fitted_params['lambda_iso'], true_params['lambda_iso'], rtol=rtol, atol=atol_diff)
    assert jnp.allclose(fitted_params['lambda_par'], true_params['lambda_par'], rtol=rtol, atol=atol_diff)
    
    # Fractions should sum to 1? Not strictly enforced in current fit unless bounded/constrained sum
    # But with noiseless data it should recover.
    assert jnp.allclose(fitted_params['partial_volume_0'], true_params['partial_volume_0'], rtol=rtol, atol=0.05)
    assert jnp.allclose(fitted_params['partial_volume_1'], true_params['partial_volume_1'], rtol=rtol, atol=0.05)
    
    # Orientation check (dot product)
    # Convert fitted spherical to cartesian
    fitted_mu = fitted_params['mu']
    f_theta, f_phi = fitted_mu[0], fitted_mu[1]
    f_vec = jnp.array([jnp.sin(f_theta)*jnp.cos(f_phi), jnp.sin(f_theta)*jnp.sin(f_phi), jnp.cos(f_theta)])
    
    true_mu = true_params['mu']
    t_theta, t_phi = true_mu[0], true_mu[1]
    t_vec = jnp.array([jnp.sin(t_theta)*jnp.cos(t_phi), jnp.sin(t_theta)*jnp.sin(t_phi), jnp.cos(t_theta)])
    
    dot = jnp.abs(jnp.dot(f_vec, t_vec))
    assert dot > 0.95 # Allow some deviation

if __name__ == "__main__":
    test_multi_compartment_fit()
