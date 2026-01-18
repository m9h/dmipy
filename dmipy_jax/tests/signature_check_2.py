
import jax.scipy.special as jsp
import jax.numpy as jnp

try:
    print("Attempting bessel_jn(0.5, v=1)...")
    val = jsp.bessel_jn(0.5, v=1)
    print(f"Success (z, v=1): {val}")
except Exception as e:
    print(f"Failed (z, v=1): {e}")
