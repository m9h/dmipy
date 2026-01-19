import sys
print("Python starting...")
try:
    import jax
    print("Jax imported")
    import jax.numpy as jnp
    print("Jax numpy imported")
    print(jnp.ones(3))
except Exception as e:
    print(f"Error: {e}")
