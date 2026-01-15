
import jax.numpy as jnp

__all__ = [
    'S2SphereStejskalTannerApproximation'
]

class S2SphereStejskalTannerApproximation:
    r"""
    The Stejskal Tanner signal approximation of a sphere model.
    It assumes that pulse length is infinitessimally small and diffusion time large
    enough so that the diffusion is completely restricted. Only depends on q-value.
    JAX implementation.

    Args:
        diameter (float): Sphere diameter in meters.

    References:
        .. [1] Balinov, Balin, et al. "The NMR self-diffusion method applied to
            restricted diffusion. Simulation of echo attenuation from molecules in
            spheres and between planes." Journal of Magnetic Resonance, Series A
            104.1 (1993): 17-25.
        :cite:p:`palombo2020sandi`
    """

    def __init__(self, diameter=None):
        self.diameter = diameter

    def sphere_attenuation(self, q, diameter):
        "The signal attenuation for the sphere model."
        radius = diameter / 2
        # Avoid division by zero by replacing q=0 with dummy value 1.0
        # The limit of the function as q->0 is 1.0, effectively handling the q=0 case.
        q_safe = jnp.where(q > 0, q, 1.0)
        
        factor = 2 * jnp.pi * q_safe * radius
        
        E = (
            3 / (factor ** 2) *
            (
                jnp.sin(factor) / factor -
                jnp.cos(factor)
            )
        ) ** 2
        
        # Return 1.0 where q was <= 0 (specifically q=0)
        return jnp.where(q > 0, E, 1.0)

    def __call__(self, qvalues, **kwargs):
        r'''
        Calculates the signal attenuation.

        Parameters
        ----------
        qvalues : jax.numpy.ndarray, shape(N),
            q-values in 1/m.
        kwargs: keyword arguments to the model parameter values,
            Is internally given as **parameter_dictionary.

        Returns
        -------
        attenuation : jax.numpy.ndarray, shape(N),
            signal attenuation
        '''
        diameter = kwargs.get('diameter', self.diameter)
        return self.sphere_attenuation(qvalues, diameter)

# Note on other models in dmipy/signal_models/sphere_models.py:
# _S3SphereCallaghanApproximation uses scipy.special.spherical_jn.
# JAX does not currently have a direct equivalent for spherical_jn in jax.scipy.special.
# It would need to be implemented using jax.scipy.special.bessel_jn(n + 0.5, z) * sqrt(pi/2z).
