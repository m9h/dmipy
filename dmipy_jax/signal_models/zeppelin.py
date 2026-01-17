from dmipy_jax.signal_models import g2_zeppelin
from jax import numpy as jnp

class Zeppelin:
    r"""
    The Zeppelin model [1]_ - an axially symmetric Tensor.
    
    Args:
        mu: (3,) Fiber orientation.
        lambda_par: Diffusivity along the axis.
        lambda_perp: Diffusivity perpendicular to the axis.
    """

    parameter_names = ['mu', 'lambda_par', 'lambda_perp']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'lambda_perp': 1}

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
        
        # Optimization: precompute/jit usually handles this, but explicit is good.
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        
        mu_cart = jnp.array([st * cp, st * sp, ct])

        return g2_zeppelin(bvals, gradient_directions, mu_cart, lambda_par, lambda_perp)
