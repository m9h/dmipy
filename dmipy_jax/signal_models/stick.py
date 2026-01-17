from dmipy_jax.signal_models import c1_stick
from jax import numpy as jnp

class Stick:
    r"""
    The Stick model [1]_ - a cylinder with zero radius.
    JAX implementation using c1_stick kernel.
    """

    parameter_names = ['mu', 'lambda_par']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1}

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)

        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        
        # Optimization: explicit conversion
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        
        mu_cart = jnp.array([st * cp, st * sp, ct])

        return c1_stick(bvals, gradient_directions, mu_cart, lambda_par)
