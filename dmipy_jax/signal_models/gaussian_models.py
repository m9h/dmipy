import jax.numpy as jnp
from jax import jit

@jit
def g1_ball(bvals, bvecs, lambda_iso):
    """
    Computes signal for an Isotropic Ball (Free Diffusion).
    
    Args:
        bvals: (N,) array.
        bvecs: (N, 3) array (ignored, but kept for API consistency).
        lambda_iso: Scalar isotropic diffusivity.
    """
    # Simple mono-exponential decay
    # S = exp(-b * d_iso)
    return jnp.exp(-bvals * lambda_iso)

@jit
def g2_zeppelin(bvals, bvecs, mu, lambda_par, lambda_perp):
    """
    Computes signal for a Zeppelin (Cylindrically Symmetric Tensor).
    
    Args:
        mu: (3,) Fiber orientation.
        lambda_par: Diffusivity along the axis.
        lambda_perp: Diffusivity perpendicular to the axis.
    """
    # Project gradients onto fiber axis
    dot_prod = jnp.dot(bvecs, mu)
    dot_prod_sq = dot_prod ** 2
    
    # S = exp( -b * (lambda_par * (g.mu)^2 + lambda_perp * (1 - (g.mu)^2)) )
    exponent = -bvals * (lambda_par * dot_prod_sq + lambda_perp * (1 - dot_prod_sq))
    return jnp.exp(exponent)
