import jax.numpy as jnp

class C1Stick:
    r"""
    The Stick model [1]_ - a cylinder with zero radius - typically used
    for intra-axonal diffusion. JAX implementation.

    Args:
        mu (jax.numpy.ndarray): Angles [theta, phi] representing the main orientation
            on the sphere.
            - theta: inclination (polar) angle [0, pi].
            - phi: azimuthal angle [-pi, pi].
            Shape: (2,).
        lambda_par (float): Parallel diffusivity in m^2/s.

    References:
        .. [1] Behrens, T. E. J., et al. "Characterization and propagation of uncertainty
            in diffusion-weighted MR imaging." Magnetic Resonance in Medicine 50.5
            (2003): 1077-1088.
        :cite:p:`zhang2012noddi`
    """

    parameter_names = ['mu', 'lambda_par']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1}

    def __init__(self, mu=None, lambda_par=None):
        self.mu = mu
        self.lambda_par = lambda_par

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
        mu = kwargs.get('mu', self.mu)

        # Convert simple [theta, phi] to cartesian
        # Assuming mu is [theta, phi]
        theta = mu[0]
        phi = mu[1]
        
        sintheta = jnp.sin(theta)
        
        # Consistent with utils.unitsphere2cart_1d
        mu_cart_0 = sintheta * jnp.cos(phi)
        mu_cart_1 = sintheta * jnp.sin(phi)
        mu_cart_2 = jnp.cos(theta)
        
        mu_cart = jnp.array([mu_cart_0, mu_cart_1, mu_cart_2])

        # Calculate attenuation
        # E = exp(-b * lambda_par * (n . mu)^2)
        dot_prod = jnp.dot(gradient_directions, mu_cart)
        E_stick = jnp.exp(-bvals * lambda_par * dot_prod ** 2)
        
        return E_stick
