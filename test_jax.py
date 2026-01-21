
import sys
print("Python starting...")
sys.stdout.flush()
try:
    import jax
    import jax.numpy as jnp
    print("JAX imported.")
    print(jax.devices())
    x = jnp.ones(5)
    print("JAX computation works:", x)
except Exception as e:
    print("JAX failed:", e)
    import traceback
    traceback.print_exc()
