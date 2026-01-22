
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from dmipy_jax.signal_models import sphere_models, plane_models

def test_dot_execution():
    model = sphere_models.S1Dot()
    bvals = jnp.ones(10)
    bvecs = jnp.zeros((10, 3))
    signal = eqx.filter_jit(model)(bvals, bvecs)
    assert jnp.all(signal == 1.0)
    assert signal.shape == (10,)

def test_sphere_stejskal_tanner_execution():
    model = sphere_models.SphereStejskalTanner()
    # Mock data
    N = 10
    bvals = jnp.linspace(0, 3000, N)
    bvecs = jnp.zeros((N, 3))
    
    # Must provide q explicitly or timing
    params = {
        'diameter': 5e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 10.0)
    assert jnp.all(signal >= 0.0)

def test_sphere_callaghan_execution():
    model = sphere_models.SphereCallaghan(number_of_roots=10, number_of_functions=10)
    N = 10
    bvals = jnp.linspace(0, 3000, N)
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'diffusion_constant': 2e-9,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 1.0 + 1e-6) # Allow numerical wiggle
    assert jnp.all(signal >= 0.0)

def test_plane_stejskal_tanner_execution():
    model = plane_models.PlaneStejskalTanner()
    N = 10
    bvals = jnp.linspace(0, 3000, N)
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = eqx.filter_jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    assert jnp.all(signal <= 10.0)
    assert jnp.all(signal >= 0.0)

def test_plane_callaghan_execution():
    model = plane_models.PlaneCallaghan(number_of_roots=20)
    N = 10
    bvals = jnp.linspace(0, 3000, N)
    bvecs = jnp.zeros((N, 3))
    
    params = {
        'diameter': 5e-6,
        'diffusion_constant': 2e-9,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    signal = jax.jit(model)(bvals, bvecs, **params)
    assert signal.shape == (N,)
    # Plane Callaghan has sines and cosines, should decay.
    assert jnp.all(jnp.isfinite(signal))

def test_sphere_equivalence_legacy():
    try:
        from dmipy.signal_models import sphere_models as leg_sphere
        from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
    except ImportError:
        pytest.skip("Legacy dmipy not found.")
        
    # Stejskal Tanner
    print("Testing Sphere Stejskal Tanner Equivalence...")
    bvals = np.array([0, 1000e6, 2000e6])
    bvecs = np.zeros((3,3)); bvecs[:,0]=1
    delta=0.01; Delta=0.03
    acq = acquisition_scheme_from_bvalues(bvals, bvecs, delta=delta, Delta=Delta)
    
    leg_model = leg_sphere.S2SphereStejskalTannerApproximation(diameter=6e-6)
    leg_sig = leg_model(acq)
    
    jax_model = sphere_models.SphereStejskalTanner(diameter=6e-6)
    jax_sig = jax_model(jnp.array(bvals), jnp.array(bvecs), big_delta=Delta, small_delta=delta)
    
    np.testing.assert_allclose(leg_sig, jax_sig, atol=1e-5)
    print("Sphere Stejskal Tanner Matches.")

def test_plane_equivalence_legacy():
    try:
        from dmipy.signal_models import plane_models as leg_plane
        from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
    except ImportError:
        pytest.skip("Legacy dmipy not found.")

    # Stejskal Tanner
    print("Testing Plane Stejskal Tanner Equivalence...")
    bvals = np.array([0, 1000e6])
    bvecs = np.zeros((2,3)); bvecs[:,0]=1
    delta=0.01; Delta=0.03
    acq = acquisition_scheme_from_bvalues(bvals, bvecs, delta=delta, Delta=Delta)
    
    leg_model = leg_plane.P2PlaneStejskalTannerApproximation(diameter=5e-6)
    leg_sig = leg_model(acq)
    
    jax_model = plane_models.PlaneStejskalTanner(diameter=5e-6)
    jax_sig = jax_model(jnp.array(bvals), jnp.array(bvecs), big_delta=Delta, small_delta=delta)
    
    np.testing.assert_allclose(leg_sig, jax_sig, atol=1e-5)
    print("Plane Stejskal Tanner Matches.")

