import jax.scipy.special as jsp
import jax.numpy as jnp
import inspect

print("Available in jsp:")
print([x for x in dir(jsp) if 'j' in x])

z = jnp.array([1.0, 5.0])

try:
    print("Testing bessel_jn(1, z)...")
    print(jsp.bessel_jn(1, z))
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing bessel_jn(z, v=1)...")
    print(jsp.bessel_jn(z, v=1))
except Exception as e:
    print(f"Failed: {e}")

try:
    print("Testing j1(z)...")
    print(jsp.j1(z))
except Exception as e:
    print(f"Failed: {e}")
