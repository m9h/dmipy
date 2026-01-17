import jax.numpy as jnp
from dmipy_jax.signal_models import c2_cylinder

# Standard HCP Protocol
big_delta = 0.0431  # 43.1 ms
small_delta = 0.0106 # 10.6 ms
tau = big_delta - small_delta / 3.0
bvals = jnp.array([1000.0, 2000.0, 3000.0])

# Gradients perpendicular to X-axis
bvecs_perp = jnp.array([[0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0]])

# Gradients parallel to X-axis
bvecs_par = jnp.array([[1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]])

mu = jnp.array([1.0, 0.0, 0.0]) # Fiber along X-axis
lambda_par = 1.7e-3 # typical axon diffusivity (mm^2/s) NOTE: dmipy usually uses m^2/s, but here we used mm^2/s input for bvals? 
# Wait, standard dmipy input: bvals in s/mm^2, diffusivities in mm^2/s?
# Or bvals in s/m^2. Let's stick to SI units or consistent units.
# HCP bvals are s/mm^2 (e.g. 1000).
# Diffusivity approx 3e-3 mm^2/s (free water) or 1.7e-3 (axon).

# Diameter 10um
# Restricted signal perpendicular to fiber should be high (restricted).
# Parallel signal should decay.

print("Test 1: Perpendicular Signal (Restricted)")
s_perp = c2_cylinder(bvals, bvecs_perp, mu, lambda_par, 10e-3, big_delta, small_delta)
print(f"B=3000 Signal: {s_perp[-1]:.4f}")

print("\nTest 2: Parallel Signal (Free-like Decay)")
s_par = c2_cylinder(bvals, bvecs_par, mu, lambda_par, 10e-3, big_delta, small_delta)
print(f"B=3000 Signal: {s_par[-1]:.4f}")

# C2 Cylinder limit cases
# Large diameter -> Signal should decay (diffraction pattern starts)
print("\nTest 3: Large Diameter (Diffraction Check)")
s_large = c2_cylinder(bvals, bvecs_perp, mu, lambda_par, 40e-3, big_delta, small_delta)
print(f"B=3000 Signal: {s_large[-1]:.4f}")
