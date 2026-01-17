
import jax
import jax.numpy as jnp
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.composer import compose_models
import numpy as np

def debug_gradients():
    # Setup
    bvalues = jnp.array([0.0] + [1000.0] * 6)
    vectors = np.random.randn(7, 3)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gradient_directions = jnp.array(vectors)
    acquisition = JaxAcquisition(bvalues=bvalues, gradient_directions=gradient_directions)
    
    stick = Stick()
    model_func = compose_models([stick])
    
    # Params: [theta, phi, lambda, frac]
    # Init guess from failure case
    init_params = jnp.array([1.1, 0.6, 1.5e-9, 0.9])
    
    # Define scalar loss for grad check (e.g. sum of signal)
    def loss_fn(p):
        sig = model_func(p, acquisition)
        return jnp.sum(sig)
    
    # Compute value and grad
    val, grads = jax.value_and_grad(loss_fn)(init_params)
    
    print(f"Value: {val}")
    print(f"Gradients: {grads}")
    
    if jnp.all(grads == 0):
        print("GRADIENTS ARE ZERO!")
    else:
        print("Gradients exist.")

if __name__ == "__main__":
    debug_gradients()
