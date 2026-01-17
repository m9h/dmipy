import jax.numpy as jnp

# The first 30 roots of the derivative of the first order Bessel function J'_3/2(x)
# These are required for the Sphere GPD model (Murday-Cotts).
# We hardcode them to avoid expensive root-finding on the GPU.
SPHERE_ROOTS = jnp.array([
    2.08157598, 5.94036999, 9.20584014, 12.404445, 15.5792364, 
    18.7426456, 21.8996965, 25.0528253, 28.203361, 31.3520917,
    34.4995149, 37.6459603, 40.7916552, 43.9367615, 47.0813974,
    50.2256516, 53.3695918, 56.5132705, 59.656729, 62.8000006,
    65.9431119, 69.0860847, 72.2289406, 75.371699, 78.5143794,
    81.656999, 84.7995738, 87.9421183, 91.0846401, 94.2271424
])

# Gyromagnetic ratio for Hydrogen (rad * Hz / T)
# Or simplified: 2.675987E8 rad/s/T. 
# In dmipy we often work with 'q' directly, but if we need G, we need Gamma.
GYRO_MAGNETIC_RATIO = 267.5152549e6