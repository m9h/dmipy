import jax.scipy.special as jsp
import jax.numpy as jnp

z = jnp.array([1.0, 5.0])
print(f"z shape: {z.shape}")

# Try v=1 keyword
try:
    res = jsp.bessel_jn(z, v=1)
    print(f"Result shape: {res.shape}")
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")
