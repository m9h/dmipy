
import jax
try:
    print("Devices:", jax.devices())
    print("Default Backend:", jax.default_backend())
except Exception as e:
    print(e)
