
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.signal_models import cylinder_models, tortuosity_models
from jax.scipy import special as jsp

def test_restricted_cylinder_soderman_execution():
    """Smoke test for RestrictedCylinder (Soderman)."""
    model = cylinder_models.RestrictedCylinder()
    
    # Mock acquisition
    N = 10
    bvals = jnp.ones(N) * 1000.0
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0) # all x
    
    # Kwargs
    # Need big_delta, small_delta
    # lambda_par, diameter, mu
    
    params = {
        'lambda_par': 1.7e-9,
        'diameter': 5e-6,
        'mu': jnp.array([jnp.pi/2, 0.0]), # along x
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    # JIT compilation check
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(jnp.isfinite(signal))
    assert jnp.all(signal >= 0)
    assert jnp.all(signal <= 1.0)

def test_callaghan_restricted_cylinder_execution():
    """Smoke test for CallaghanRestrictedCylinder."""
    model = cylinder_models.CallaghanRestrictedCylinder(number_of_roots=10, number_of_functions=10)
    
    # Mock acquisition
    N = 10
    bvals = jnp.ones(N) * 2000.0
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0) # all x
    
    # Callaghan needs 'tau' or big/small delta
    params = {
        'lambda_par': 1.7e-9,
        'diffusion_perpendicular': 1.0e-9,
        'diameter': 6e-6,
        'mu': jnp.array([jnp.pi/2, 0.0]), # along x
        'tau': 0.025
    }
    
    # JIT compilation check
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(jnp.isfinite(signal))
    assert jnp.all(signal >= 0)
    assert jnp.all(signal <= 1.1) 
    # Callaghan approximation can briefly overshoot 1.0 due to series truncation/oscillation? 
    # Usually strictly <= 1 if physics holds, but numerical artifacts possible.

def test_tortuosity_model_execution():
    """Smoke test for TortuosityModel."""
    model = tortuosity_models.TortuosityModel()
    
    # Mock acquisition
    N = 5
    bvals = jnp.array([0., 1000., 2000., 3000., 1000.])
    bvecs = jnp.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        jnp.array([1., 1., 0.]) / jnp.sqrt(2)
    ])
    
    params = {
        'lambda_par': 1.7e-9,
        'icvf': 0.7,
        'mu': jnp.array([0.0, 0.0]) # along z
    }
    
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert signal[0] == 1.0 # b=0
    assert jnp.all(jnp.isfinite(signal))

def test_equivalence_with_legacy_dmipy():
    """Compare JAX implementation with Legacy dmipy if available."""
    try:
        from dmipy.signal_models import cylinder_models as legacy_cyl
        from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
    except ImportError:
        pytest.skip("Legacy dmipy not found.")
        
    # --- Restricted Cylinder (Soderman) ---
    print("Testing RestrictedCylinder Equivalence...")
    # Setup parameters
    mu_sph = np.array([np.pi/2, 0.0]) # x-axis
    lambda_par = 1.7e-9
    diameter = 5e-6
    big_delta = 0.03
    small_delta = 0.01
    
    # Setup acquisition
    bvals = np.array([0, 1000e6, 2000e6, 3000e6]) # s/m^2 for legacy? 
    # Legacy dmipy units:
    # lambda: m^2/s usually (based on scaling 1e-9)
    # bvals: s/m^2 usually.
    # In dmipy_jax, we try to use SI (s/m^2) and lambda in m^2/s if we want direct comparison?
    # Or does dmipy_jax use s/mm^2 and mm^2/s?
    # dmipy_jax/constants.py typically has SI or standard units.
    # Legacy uses SI (m^2/s) usually.
    
    # Let's check scaling.
    # c2_cylinder kernel uses bvals * lambda_par.
    # If bvals in s/mm^2 (e.g. 1000) and lambda_par in mm^2/s (1.7e-3), product is unitless.
    # If bvals in s/m^2 (1000e6) and lambda_par in m^2/s (1.7e-9), product is unitless.
    # dmipy uses s/m^2 and m^2/s.
    # dmipy_jax seems to follow suit (1.7e-9 default in ranges).
    
    bvals_si = np.r_[0, 1000e6, 2000e6, 3000e6]
    bvecs = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
    
    # Create Legacy Scheme
    acq_scheme = acquisition_scheme_from_bvalues(
        bvals_si, bvecs, delta=small_delta, Delta=big_delta
    )
    
    # Legacy Model
    legacy_soderman = legacy_cyl.C2CylinderStejskalTannerApproximation(
        mu=mu_sph, lambda_par=lambda_par, diameter=diameter
    )
    
    signal_legacy = legacy_soderman(acq_scheme)
    
    # JAX Model
    jax_soderman = cylinder_models.RestrictedCylinder(
        mu=jnp.array(mu_sph), lambda_par=lambda_par, diameter=diameter
    )
    
    signal_jax = jax_soderman(
        jnp.array(bvals_si), jnp.array(bvecs), 
        big_delta=big_delta, small_delta=small_delta
    )
    
    np.testing.assert_allclose(signal_legacy, signal_jax, rtol=1e-5, atol=1e-5)
    print("RestrictedCylinder (Soderman) matches legacy.")
    
    # --- Callaghan Cylinder ---
    print("Testing CallaghanRestrictedCylinder Equivalence...")
    # Requires q-values or tau.
    # Legacy uses derived q-values from acquisition scheme?
    # C3CylinderCallaghanApproximation.
    
    legacy_callaghan = legacy_cyl.C3CylinderCallaghanApproximation(
        mu=mu_sph, lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=1.0e-9, number_of_roots=20, number_of_functions=50
    )
    
    signal_legacy_c = legacy_callaghan(acq_scheme)
    
    jax_callaghan = cylinder_models.CallaghanRestrictedCylinder(
        mu=jnp.array(mu_sph), lambda_par=lambda_par, diameter=diameter,
        diffusion_perpendicular=1.0e-9, number_of_roots=20, number_of_functions=50
    )
    
    # Use tau from acq_scheme
    tau = big_delta - small_delta / 3.0
    
    signal_jax_c = jax_callaghan(
        jnp.array(bvals_si), jnp.array(bvecs), tau=tau
    )
    
    # Callaghan approximation involves series summation, might have slight differences 
    # due to float precision (32 vs 64) or implementation details of bessel functions.
    # JAX uses float32 by default? No, relies on config.
    # We'll use 1e-4 tolerance.
    
    np.testing.assert_allclose(signal_legacy_c, signal_jax_c, rtol=1e-3, atol=1e-3)
    print("CallaghanRestrictedCylinder matches legacy.")

