from dmipy_jax.signal_models import c1_stick, c2_cylinder
from jax import numpy as jnp

class C1Stick:
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
        
        mu_cart = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])

        return c1_stick(bvals, gradient_directions, mu_cart, lambda_par)


class C2Cylinder:
    r"""
    The Cylinder model [1]_ - finite radius with Soderman approximation.
    JAX implementation using c2_cylinder kernel.
    """

    parameter_names = ['mu', 'lambda_par', 'diameter']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'diameter': 1}

    def __init__(self, mu=None, lambda_par=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

    def __call__(self, bvals, gradient_directions, **kwargs):
        # Requires big_delta and small_delta from acquisition scheme/kwargs?
        # The kernel c2_cylinder needs big_delta, small_delta. 
        # These are typically in kwargs if passed from acquisition.
        # But 'compose_models' in composer.py passes bvals, gradient_directions explicitly,
        # and relies on kwargs to carry other things.
        
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        diameter = kwargs.get('diameter', self.diameter)
        mu = kwargs.get('mu', self.mu)
        
        # Acquisition parameters
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        
        # If not provided, raise error or assume simplistic case?
        # For now, we assume they are passed.
        
        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        
        mu_cart = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])
        
        return c2_cylinder(bvals, gradient_directions, mu_cart, lambda_par, diameter, big_delta, small_delta)
