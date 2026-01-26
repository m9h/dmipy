import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder

def debug():
    print("Isolating RestrictedCylinder NaNs...")
    
    # Setup inputs
    bvals = jnp.array([1000.0, 2000.0]) * 1e9 # s/m^2
    # Random bvecs
    bvecs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # Params from failed run (SI)
    mu_sph = jnp.array([2.2404916286468506, 1.4304563999176025])
    lambda_par = 2.2102262242640336e-09
    diameter = 1.4580517017748207e-05
    
    kwargs = {
        'big_delta': 0.02,
        'small_delta': 0.01,
        'delta': 0.01,
        'Delta': 0.02
    }
    
    model = RestrictedCylinder(mu=mu_sph, lambda_par=lambda_par, diameter=diameter)
    
    print("Running Forward Pass with SI units...")
    try:
        sig = model(bvals, bvecs, **kwargs)
        print("Signal:", sig)
        if jnp.isnan(sig).any():
            print("NaN detected!")
    except Exception as e:
        print(f"Exception: {e}")
        
    # Check Scipy
    try:
        import scipy.special
        print("Scipy Special J1(56000):", scipy.special.j1(56000.0))
        print("Scipy Special J1(56000) float32:", scipy.special.j1(np.array(56000.0, dtype=np.float32)))
    except ImportError:
        print("Scipy not installed!")

if __name__ == "__main__":
    debug()
