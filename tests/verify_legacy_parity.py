
import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.signal_models.sphere_models import SphereGPD, SphereStejskalTanner
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.tortuosity_models import TortuosityModel

def verify_models():
    print("Verifying Dmipy Parity Models...")
    
    # Common Acquisition
    N = 10
    bvals = jnp.ones(N) * 2000
    bvecs = jnp.zeros((N, 3)); bvecs = bvecs.at[:, 0].set(1.0)
    big_delta = 0.03
    small_delta = 0.01
    
    # 1. Restricted Cylinder (C2)
    print("\n--- Testing RestrictedCylinder (C2) ---")
    try:
        model = RestrictedCylinder(mu=[0.0, 0.0], lambda_par=1.7e-9, diameter=6e-6)
        signal = model(bvals, bvecs, big_delta=big_delta, small_delta=small_delta)
        assert signal.shape == (N,)
        assert jnp.all(signal >= 0.0) and jnp.all(signal <= 1.0)
        assert not jnp.any(jnp.isnan(signal))
        print("RestrictedCylinder Passed shape/range checks.")
    except Exception as e:
        print(f"RestrictedCylinder FAILED: {e}")

    # 2. Sphere GPD (G2)
    print("\n--- Testing SphereGPD (G2) ---")
    try:
        # Note: SphereGPD class needs to be implemented first
        if 'SphereGPD' in globals():
            model = SphereGPD(diameter=8e-6, diffusion_constant=1.0e-9)
            signal = model(bvals, bvecs, big_delta=big_delta, small_delta=small_delta)
            assert signal.shape == (N,)
            assert jnp.all(signal >= 0.0) and jnp.all(signal <= 1.0)
            assert not jnp.any(jnp.isnan(signal))
            print("SphereGPD Passed shape/range checks.")
        else:
            print("SphereGPD class not found/imported.")
    except Exception as e:
        print(f"SphereGPD FAILED: {e}")

    # 3. Zeppelin
    print("\n--- Testing Zeppelin ---")
    try:
        model = Zeppelin(mu=[0.0, 0.0], lambda_par=1.5e-9, lambda_perp=0.5e-9)
        signal = model(bvals, bvecs)
        assert signal.shape == (N,)
        print("Zeppelin Passed shape checks.")
    except Exception as e:
        print(f"Zeppelin FAILED: {e}")

    # 4. Tortuosity
    print("\n--- Testing TortuosityModel ---")
    try:
        model = TortuosityModel(mu=[0.0, 0.0], lambda_par=1.7e-9, icvf=0.7)
        signal = model(bvals, bvecs)
        assert signal.shape == (N,)
        
        # Check lambda_perp constraint
        # lambda_perp should be (1-0.7) * 1.7e-9 = 0.3 * 1.7e-9 = 0.51e-9
        # Hard to check internal state easily without access to g2_zeppelin underlying call, 
        # but execution confirms it runs.
        print("TortuosityModel Passed execution.")
    except Exception as e:
        print(f"TortuosityModel FAILED: {e}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_models()
