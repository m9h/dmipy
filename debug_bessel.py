import jax.scipy.special as jsp
import jax.numpy as jnp

try:
    print(f"bessel_jn doc: {jsp.bessel_jn.__doc__}")
except:
    pass

try:
    print("Trying bessel_jn(2.0, v=1, n_iter=50)")
    res = jsp.bessel_jn(2.0, v=1, n_iter=50)
    print(f"Result shape: {res.shape}")
    print(f"J0: {res[0]}")
    print(f"J1: {res[1]}")
    
    import scipy.special as ssp
    print(f"Scipy J1(2.0): {ssp.j1(2.0)}")
except Exception as e:
    print(f"Failed (2.0, v=1): {e}")

import jax.lax
print("\nChecking jax.lax:")
for name in dir(jax.lax):
    if 'bessel' in name:
        print(name)
