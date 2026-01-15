import jax.numpy as jnp

class G1Ball:
    r"""
    The Ball model [1]_ - an isotropic Tensor with one diffusivity.
    JAX implementation.

    Args:
        lambda_iso (float): Isotropic diffusivity in m^2/s.

    References:
        .. [1] Behrens, T. E. J., et al.
           "Characterization and propagation of uncertainty in
            diffusion-weighted MR imaging"
            Magnetic Resonance in Medicine (2003)
    """


    parameter_names = ['lambda_iso']
    parameter_cardinality = {'lambda_iso': 1}

    def __init__(self, lambda_iso=None):
        self.lambda_iso = lambda_iso

    def __call__(self, bvals, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        bvals : jax.numpy.ndarray, shape(N),
            b-values.
        kwargs: keyword arguments to the model parameter values.
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : jax.numpy.ndarray, shape(N),
            signal attenuation
        '''
        lambda_iso = kwargs.get('lambda_iso', self.lambda_iso)
        E_ball = jnp.exp(-bvals * lambda_iso)
        return E_ball


class G2Zeppelin:
    r"""
    The Zeppelin model [1]_ - an axially symmetric Tensor - typically used
    for extra-axonal diffusion. JAX implementation.

    Args:
        mu (jax.numpy.ndarray): Angles [theta, phi] representing the main orientation
            on the sphere.
            - theta: inclination (polar) angle [0, pi].
            - phi: azimuthal angle [-pi, pi].
            Shape: (2,).
        lambda_par (float): Parallel diffusivity in m^2/s.
        lambda_perp (float): Perpendicular diffusivity in m^2/s.

    References:
        .. [1] Panagiotaki et al. "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """

    parameter_names = ['mu', 'lambda_par', 'lambda_perp']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'lambda_perp': 1}

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, bvals, gradient_directions, **kwargs):
        r'''
        Estimates the signal attenuation.

        Parameters
        ----------
        bvals : jax.numpy.ndarray, shape(N),
            b-values.
        gradient_directions : jax.numpy.ndarray, shape(N, 3),
            gradient directions (n).
        kwargs: keyword arguments to the model parameter values.

        Returns
        -------
        attenuation : jax.numpy.ndarray, shape(N),
            signal attenuation
        '''
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)

        # Convert [theta, phi] to cartesian
        theta = mu[0]
        phi = mu[1]
        
        sintheta = jnp.sin(theta)
        mu_cart_0 = sintheta * jnp.cos(phi)
        mu_cart_1 = sintheta * jnp.sin(phi)
        mu_cart_2 = jnp.cos(theta)
        
        mu_cart = jnp.array([mu_cart_0, mu_cart_1, mu_cart_2])

        # Calculate attenuation
        # E = exp(-b * (lambda_par * (n.mu)^2 + lambda_perp * (1 - (n.mu)^2)))
        
        dot_prod = jnp.dot(gradient_directions, mu_cart)
        dot_prod_sq = dot_prod ** 2
        
        exponent = -bvals * (lambda_par * dot_prod_sq + lambda_perp * (1 - dot_prod_sq))
        E_zeppelin = jnp.exp(exponent)
        
        return E_zeppelin
