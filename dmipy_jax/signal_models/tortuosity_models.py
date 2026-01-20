from dmipy_jax.signal_models.gaussian_models import g2_zeppelin
from jax import numpy as jnp


class TortuosityModel:
    r"""
    A Tortuosity constrained Zeppelin model.
    The perpendicular diffusivity is constrained by the parallel diffusivity and the 
    intra-cellular volume fraction (icvf) using the tortuosity approximation [1, 2].
    
    lambda_perp = (1 - icvf) * lambda_par

    This model represents the EXTRA-cellular compartment signal, but parameterized by 
    intra-cellular properties to enforce the tortuosity constraint.

    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in mm^2/s.
    icvf : float
        intra-cellular volume fraction [0, 1].

    References
    ----------
    .. [1] Szafer et al. "Theoretical model for water diffusion in
            tissues." Magnetic resonance in medicine 33.5 (1995): 697-712.
    .. [2] Zhang et al. "NODDI: handy geometric modeling of the human brain
            connectome with in vivo MRI." Neuroimage 61.4 (2012): 1000-1016.
    """

    parameter_names = ['mu', 'lambda_par', 'icvf']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'icvf': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'icvf': (0.1, 0.99)
    }

    def __init__(self, mu=None, lambda_par=None, icvf=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.icvf = icvf

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        icvf = kwargs.get('icvf', self.icvf)
        mu = kwargs.get('mu', self.mu)
        
        # Calculate lambda_perp using tortuosity constraint
        # T1 tortuosity: lambda_perp = (1 - icvf) * lambda_par
        # Note: In standard NODDI, it's (1-ficvf) * dpar, assuming only intra+extra.
        # If there is also ISO, it gets more complex, but local tortuosity is usually just 1-ficvf.
        
        lambda_perp = (1.0 - icvf) * lambda_par

        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        
        mu_cart = jnp.array([st * cp, st * sp, ct])

        return g2_zeppelin(bvals, gradient_directions, mu_cart, lambda_par, lambda_perp)
