from dmipy_jax.signal_models import g1_ball, g2_zeppelin, g2_tensor
from jax import numpy as jnp

class G1Ball:
    r"""
    The Ball model [1]_ - an isotropic Tensor with one diffusivity.
    JAX implementation.

    Args:
        lambda_iso (float): Isotropic diffusivity in m^2/s.
    """

    parameter_names = ['lambda_iso']
    parameter_cardinality = {'lambda_iso': 1}
    parameter_ranges = {
        'lambda_iso': (0.1e-9, 3e-9)
    }


    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, bvals, **kwargs):
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        return g1_ball(bvals, None, lambda_iso)


class G2Zeppelin:
    r"""
    The Zeppelin model [1]_ - an axially symmetric Tensor.
    """

    parameter_names = ['mu', 'lambda_par', 'lambda_perp']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'lambda_perp': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'lambda_perp': (0.1e-9, 3e-9)
    }


    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)

        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        mu_cart = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])

        return g2_zeppelin(bvals, gradient_directions, mu_cart, lambda_par, lambda_perp)


class G2Tensor:
    r"""
    The Full Tensor model [1]_ - a general anisotropic diffusion tensor.
    Parametrized by 3 eigenvalues and orientation (Euler angles or 2 vectors).
    Here we use 3 eigenvalues + 3 Euler angles (alpha, beta, gamma) for full rotation.
    
    Self-Correction: dmipy typically uses eigenvectors or Euler angles. 
    Optimization-wise, Euler angles (3 params) map to a rotation matrix R.
    R * [1,0,0], R * [0,1,0] give e1, e2.
    """

    parameter_names = ['eigenvalues', 'orientation']
    # 3 eigenvalues, 3 Euler angles for orientation
    parameter_cardinality = {'eigenvalues': 3, 'orientation': 3}

    def __init__(self, eigenvalues=None, orientation=None):
        self.eigenvalues = eigenvalues
        self.orientation = orientation

    def __call__(self, bvals, gradient_directions, **kwargs):
        eigenvalues = kwargs.get('eigenvalues', self.eigenvalues)
        orientation = kwargs.get('orientation', self.orientation)
        
        l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
        
        # Convert Euler angles (alpha, beta, gamma) to Rotation Matrix
        # Using ZYZ convention or similar typical for DTI
        # For simplicity/robustness, let's assume 'orientation' = [alpha, beta, gamma]
        # and construct R.
        
        alpha, beta, gamma = orientation[0], orientation[1], orientation[2]
        
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        cb, sb = jnp.cos(beta),  jnp.sin(beta)
        cc, sc = jnp.cos(gamma), jnp.sin(gamma)
        
        # Z1Y2Z3 intrinsic rotation or similar. 
        # Detailed implementation of specific convention needed.
        # Let's use a standard implementation: Rz(alpha) * Ry(beta) * Rz(gamma)
        
        R_alpha = jnp.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        R_beta  = jnp.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        R_gamma = jnp.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
        
        R = R_alpha @ R_beta @ R_gamma
        
        # e1, e2, e3 are columns of R
        e1 = R[:, 0]
        e2 = R[:, 1]
        # e3 is implicitly R[:, 2]
        
        return g2_tensor(bvals, gradient_directions, l1, l2, l3, e1, e2)
