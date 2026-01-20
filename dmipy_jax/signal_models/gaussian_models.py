import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx
from typing import Any

@jit
def g1_ball(bvals, bvecs, lambda_iso):
    """
    Computes signal for an Isotropic Ball (Free Diffusion).
    
    Args:
        bvals: (N,) array.
        bvecs: (N, 3) array (ignored, but kept for API consistency).
        lambda_iso: Scalar isotropic diffusivity.
    """
    # Debug
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


class Ball:
    """
    Isotropic Ball model class.
    """
    parameter_names = ['lambda_iso']
    parameter_cardinality = {'lambda_iso': 1}
    parameter_ranges = {'lambda_iso': (0.0, 3e-9)}

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        return g1_ball(bvals, gradient_directions, lambda_iso)


class Tensor(eqx.Module):
    r"""
    The full Gaussian Tensor model (Ellipsoid) [1]_.
    
    Parameters
    ----------
    lambda_1 : float
        First eigenvalue (diffusivity) in m^2/s.
    lambda_2 : float
        Second eigenvalue in m^2/s.
    lambda_3 : float
        Third eigenvalue in m^2/s.
    alpha : float
        First Euler angle (Z-Y-Z convention) in radians.
    beta : float
        Second Euler angle (Z-Y-Z convention) in radians.
    gamma : float
        Third Euler angle (Z-Y-Z convention) in radians.
        
    References
    ----------
    .. [1] Basser, Peter J., James Mattiello, and Denis LeBihan.
           "MR diffusion tensor spectroscopy and imaging."
           Biophysical journal 66.1 (1994): 259-267.
    """
    
    lambda_1: Any = None
    lambda_2: Any = None
    lambda_3: Any = None
    alpha: Any = None
    beta: Any = None
    gamma: Any = None

    parameter_names = ('lambda_1', 'lambda_2', 'lambda_3', 'alpha', 'beta', 'gamma')
    parameter_cardinality = {
        'lambda_1': 1, 'lambda_2': 1, 'lambda_3': 1,
        'alpha': 1, 'beta': 1, 'gamma': 1
    }
    parameter_ranges = {
        'lambda_1': (0.1e-9, 3e-9),
        'lambda_2': (0.1e-9, 3e-9),
        'lambda_3': (0.1e-9, 3e-9),
        'alpha': (-jnp.pi, jnp.pi),
        'beta': (0, jnp.pi),
        'gamma': (-jnp.pi, jnp.pi)
    }

    def __init__(self, lambda_1=None, lambda_2=None, lambda_3=None, 
                 alpha=None, beta=None, gamma=None):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, bvals, gradient_directions, **kwargs):
        l1 = kwargs.get('lambda_1', self.lambda_1)
        l2 = kwargs.get('lambda_2', self.lambda_2)
        l3 = kwargs.get('lambda_3', self.lambda_3)
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        gamma = kwargs.get('gamma', self.gamma)
        
        # Convert Euler angles (Z-Y-Z) to Rotation Matrix
        # R = Rz(alpha) * Ry(beta) * Rz(gamma)
        # Note: Dmipy and standard DWI often use Z-Y-Z
        
        ca = jnp.cos(alpha)
        sa = jnp.sin(alpha)
        cb = jnp.cos(beta)
        sb = jnp.sin(beta)
        cg = jnp.cos(gamma)
        sg = jnp.sin(gamma)
        
        # R11 = ca cb cg - sa sg
        # R12 = -ca cb sg - sa cg
        # R13 = ca sb
        # R21 = sa cb cg + ca sg
        # R22 = -sa cb sg + ca cg
        # R23 = sa sb
        # R31 = -sb cg
        # R32 = sb sg
        # R33 = cb
        
        # Eigenvectors are columns of R
        e1 = jnp.array([ca*cb*cg - sa*sg, sa*cb*cg + ca*sg, -sb*cg])
        e2 = jnp.array([-ca*cb*sg - sa*cg, -sa*cb*sg + ca*cg, sb*sg])
        # e3 = jnp.array([ca*sb, sa*sb, cb]) # Not needed for kernel but check g2_tensor
        
        # Ensure eigenvectors are correct shape (3,) or broadcastable if inputs are arrays
        # If inputs are arrays (N,), e1 will be (3, N) if we stack like this?
        # Actually if alpha is (N,), then ca is (N,).
        # jnp.array([...]) results in (3, N).
        # g2_tensor expects bvecs (N, 3). e1 should be (N, 3) or (3,)
        # If we return (3, N), we need to transpose to (N, 3) for dot product with bvecs.
        
        if jnp.ndim(ca) > 0:
            e1 = jnp.moveaxis(e1, 0, -1) # (..., 3)
            e2 = jnp.moveaxis(e2, 0, -1)
        
        # Kernel g2_tensor calculates e3 internally via cross product
        return g2_tensor(bvals, gradient_directions, l1, l2, l3, e1, e2)
