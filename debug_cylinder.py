
import jax
import jax.numpy as jnp
from dmipy_jax.signal_models import cylinder_models

def test_cylinder_standalone():
    cylinder = cylinder_models.RestrictedCylinder()
    
    N = 2
    bvals = jnp.ones(N) * 1000.0
    bvecs = jnp.zeros((N, 3))
    bvecs = bvecs.at[:, 0].set(1.0)
    
    params = {
        'lambda_par': 1.7e-9,
        'mu': jnp.array([jnp.pi/2, 0.0]),
        'diameter': 5e-6,
        'big_delta': 0.03,
        'small_delta': 0.01
    }
    
    res = cylinder(bvals, bvecs, **params)
    print("Standalone result shape:", res.shape)
    
if __name__ == "__main__":
    test_cylinder_standalone()
