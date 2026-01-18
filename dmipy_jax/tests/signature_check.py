
import jax.scipy.special as jsp
import jax.numpy as jnp

try:
    print("Attempting bessel_jn(1, 0.5)...")
    val = jsp.bessel_jn(1, 0.5)
    print(f"Success 2 args: {val}")
except Exception as e:
    print(f"Failed 2 args: {e}")

try:
    print("Attempting bessel_jn(0.5)...")
    val = jsp.bessel_jn(0.5)
    print(f"Success 1 arg: {val}")
except Exception as e:
    print(f"Failed 1 arg: {e}")
