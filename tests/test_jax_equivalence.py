
import numpy as np
import jax
import jax.numpy as jnp
import numpy.testing as npt
from dmipy.signal_models.cylinder_models import C1Stick as OriginalStick
from dmipy_jax.cylinder import C1Stick as JaxStick
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.utils import utils

def test_stick_equivalence():
    # Setup parameters
    # User requested mu = [1, 0, 0].
    # C1Stick expects mu as [theta, phi] (spherical coordinates).
    # [1, 0, 0] (Cartesian) -> theta=pi/2, phi=0.
    mu_cart = np.array([1., 0., 0.])
    mu_spherical = utils.cart2sphere(mu_cart)[1:] # returns [r, theta, phi], we need [theta, phi]
    
    lambda_par = 1.7e-9  # SI units
    
    # Mock Acquisition Scheme
    # 30 gradient directions, b=1000 scaled to SI
    N_meas = 30
    bval = 1000e6 # s/m^2
    
    bvalues = np.tile(bval, N_meas) 
    
    # Random gradient directions
    np.random.seed(42)
    gradient_directions = np.random.randn(N_meas, 3)
    gradient_directions /= np.linalg.norm(gradient_directions, axis=1)[:, None]
    
    # Create dmipy acquisition scheme
    acq_scheme = acquisition_scheme_from_bvalues(
        bvalues=bvalues,
        gradient_directions=gradient_directions,
        delta=0.01,
        Delta=0.02
    )
    
    # Instantiate Models
    orig_model = OriginalStick()
    jax_model = JaxStick()
    
    # Prepare inputs
    params = {
        'mu': mu_spherical,
        'lambda_par': lambda_par
    }
    
    # Prediction: Original
    if hasattr(orig_model, 'predict'):
        orig_signal = orig_model.predict(acq_scheme, **params)
    else:
        orig_signal = orig_model(acq_scheme, **params)
        
    # Prediction: JAX
    bvals_jax = jnp.array(bvalues)
    grads_jax = jnp.array(gradient_directions)
    
    # The JAX model __call__ expects bvals, gradient_directions.
    # We pass params as kwargs.
    jax_output_jax = jax_model(bvals_jax, grads_jax, **params)
    
    # Block until ready and convert to numpy
    jax_signal = np.array(jax_output_jax.block_until_ready())
    
    # Compare
    try:
        npt.assert_allclose(orig_signal, jax_signal, rtol=1e-5)
        print("Test PASSED: JAX and Original Stick models match.")
    except AssertionError as e:
        diff = np.abs(orig_signal - jax_signal)
        print(f"Test FAILED. Max difference: {np.max(diff)}")
        print(f"Original: {orig_signal[:5]}")
        print(f"JAX: {jax_signal[:5]}")
        raise e

if __name__ == "__main__":
    test_stick_equivalence()
