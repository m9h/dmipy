import jax.numpy as jnp
from jax import jit

@jit
def c1_stick(bvals, bvecs, mu, lambda_par):
    """
    Computes signal for a Stick (zero-radius cylinder).
    
    Args:
        bvals: (N,) array of b-values in s/mm^2.
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (mm^2/s).
        
    Returns:
        (N,) array of signal attenuation (0.0 to 1.0).
    """
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    
    # Signal decay depends only on the parallel component
    # S = exp(-b * d_par * (g . mu)^2)
    signal = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    return signal
