
import jax.scipy.special as jsp
import jax.numpy as jnp
import inspect

print("Checking jsp.bessel_jn signature...")
try:
    print(inspect.signature(jsp.bessel_jn))
except Exception as e:
    print(f"inspect failed: {e}")

try:
    z = jnp.array([1.0, 2.0])
    v = 1
    print("Trying jsp.bessel_jn(v, z)...")
    res = jsp.bessel_jn(v, z)
    print("Success jsp.bessel_jn(v, z)")
except Exception as e:
    print(f"Failed jsp.bessel_jn(v, z): {e}")

try:
    z = jnp.array([1.0, 2.0])
    v = 1
    print("Trying jsp.bessel_jn(z, v)...")
    res = jsp.bessel_jn(z, v)
    print("Success jsp.bessel_jn(z, v)") 
except Exception as e:
    print(f"Failed jsp.bessel_jn(z, v): {e}")
    
try:
    z = jnp.array([1.0, 2.0])
    print("Trying jsp.bessel_jn(z, v=1)...")
    res = jsp.bessel_jn(z, v=1)
    print("Success jsp.bessel_jn(z, v=1)")
except Exception as e:
    print(f"Failed jsp.bessel_jn(z, v=1): {e}")
