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

@jit
def g2_tensor(bvals, bvecs, lambda_1, lambda_2, lambda_3, e1, e2):
    """
    Computes signal for a full Gaussian Diffusion Tensor (Ellipsoid).
    
    Args:
        lambda_1, lambda_2, lambda_3: Eigensystem eigenvalues.
        e1, e2: First two eigenvectors (e3 is cross product).
                Vectors typically defined on S2 (unit vectors).
    """
    # Construct rotation matrix R = [e1, e2, e3]
    # e3 = e1 x e2
    e3 = jnp.cross(e1, e2)
    
    # Diagonal diffusion matrix in eigenframe
    # D_diag = diag([lambda_1, lambda_2, lambda_3])
    
    # We want q^T D q.
    # q in lab frame. q_eigen = R^T q.
    # exponent = -b * g^T D_lab g
    # D_lab = R D_diag R^T
    
    # Let's project gradient onto eigenvectors
    # g_e1 = g . e1
    # g_e2 = g . e2
    # g_e3 = g . e3
    
    g_dot_e1 = jnp.dot(bvecs, e1)
    g_dot_e2 = jnp.dot(bvecs, e2)
    g_dot_e3 = jnp.dot(bvecs, e3)
    
    # Signal = exp( -b * (lambda_1 * (g.e1)^2 + lambda_2 * (g.e2)^2 + lambda_3 * (g.e3)^2) )
    exponent = -bvals * (
        lambda_1 * g_dot_e1**2 + 
        lambda_2 * g_dot_e2**2 + 
        lambda_3 * g_dot_e3**2
    )
    return jnp.exp(exponent)
