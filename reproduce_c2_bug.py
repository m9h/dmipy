import jax
import jax.numpy as jnp
from dmipy_jax.signal_models.cylinder_models import c2_cylinder

def test_c2_bug():
    # Setup inputs
    N = 10
    bvals = jnp.ones(N) * 1000
    bvecs = jnp.zeros((N, 3)); bvecs = bvecs.at[:, 0].set(1.0)
    mu = jnp.array([0.0, 1.0, 0.0]) # Perpendicular
    lambda_par = 1.7e-9
    diameter = 5e-6
    big_delta = 0.02
    small_delta = 0.01
    
    # Run
    try:
        signal = c2_cylinder(bvals, bvecs, mu, lambda_par, diameter, big_delta, small_delta)
        print("Signal shape:", signal.shape)
        print("Signal values:", signal)
        
        if signal.shape != (N,):
             print("FAILURE: Signal shape mismatch. Expected (N,), got", signal.shape)
        else:
             print("SUCCESS: Shape check passed.")
             
    except Exception as e:
        print("CRASHED:", e)

if __name__ == "__main__":
    test_c2_bug()
