import jax
import blackjax
import jax.numpy as jnp
print(f"JAX Version: {jax.__version__}")
print(f"BlackJAX Version: {blackjax.__version__}")
try:
    print(f"Backend: {jax.lib.xla_bridge.get_backend().platform}")
except:
    print(f"Backend: {jax.extend.backend.get_backend().platform}") # Fallback for newer JAX if xla_bridge is deprecated
print(f"Device Count: {jax.device_count()}")
# Test Basic Math
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print("Matrix multiplication successful.")
# Test Sampler Import (Critical for BlackJAX)
from blackjax import nuts
print("BlackJAX NUTS sampler imported successfully.")
