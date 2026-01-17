import jax.numpy as jnp
from dmipy_jax.signal_models import g3_sphere

# Standard HCP Protocol
big_delta = 0.0431  # 43.1 ms
small_delta = 0.0106 # 10.6 ms
bvals = jnp.array([1000.0, 2000.0, 3000.0])
bvecs = jnp.zeros((3, 3)) # Ignored

# Test 1: Small Soma (10 micron) - Should have signal near 1.0 (restricted)
s1 = g3_sphere(bvals, bvecs, 10e-3, 3.0e-3, big_delta, small_delta)
print(f"10um Soma Signal (b=3000): {s1[-1]:.4f}")

# Test 2: Huge Soma (100 micron) - Should decay like free water
s2 = g3_sphere(bvals, bvecs, 100e-3, 3.0e-3, big_delta, small_delta)
print(f"100um Soma Signal (b=3000): {s2[-1]:.4f}")