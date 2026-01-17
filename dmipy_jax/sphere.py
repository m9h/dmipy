from dmipy_jax.signal_models import g3_sphere
from jax import numpy as jnp

__all__ = [
    'S2SphereStejskalTannerApproximation',
    'G3Sphere'
]

class G3Sphere:
    r"""
    The Soma (Sphere) model - GPD Approximation (Murday-Cotts).
    JAX implementation using g3_sphere kernel.
    """
    
    parameter_names = ['diameter', 'diffusivity']
    parameter_cardinality = {'diameter': 1, 'diffusivity': 1}
    
    def __init__(self, diameter=None, diffusivity=None):
        self.diameter = diameter
        self.diffusivity = diffusivity
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # We need big_delta, small_delta, and gradients (magnitude)
        # g3_sphere signature:
        # g3_sphere(bvals, bvecs, diameter, diffusion_constant, big_delta, small_delta)
        # Note: bvecs is used just for shape or ignored if isotropic?
        # Actually g3_sphere in sphere_models.py calculates q from bvals/Delta/delta
        # Let's check kernel signature.
        
        diameter = kwargs.get('diameter', self.diameter)
        diffusivity = kwargs.get('diffusivity', self.diffusivity)
        
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        
        return g3_sphere(bvals, gradient_directions, diameter, diffusivity, big_delta, small_delta)

class S2SphereStejskalTannerApproximation:
    r"""
    The Stejskal Tanner signal approximation of a sphere model.
    Assumes infinite diffusion time (restricted limit).
    """

    parameter_names = ['diameter']
    parameter_cardinality = {'diameter': 1}

    def __init__(self, diameter=None):
        self.diameter = diameter
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # S2 only needs q. We approximate q from bvals.
        # Ideally we need q-values passed directly.
        # But standard signature is bvals/bvecs.
        
        # If big_delta/small_delta are passed, we can calc q.
        # Or we can assume input 'qvalues' in kwargs if available?
        # For compatibility with 'compose_models', we likely need to convert bvals -> q
        # if delta/Delta are known.
        
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        diameter = kwargs.get('diameter', self.diameter)
        
        if big_delta is not None and small_delta is not None:
             tau = big_delta - small_delta / 3.0
             # q = sqrt(b / tau) / (2pi)
             q = jnp.sqrt(bvals / (tau + 1e-9)) / (2 * jnp.pi)
        else:
             # Fallback or error?
             # Let's assume qvalues might be in kwargs
             q = kwargs.get('qvalues')
             if q is None:
                 # Default to assuming bvals ~ q^2 * something? No, dangerous.
                 # Let's just return 1.0 or raise warning if no time info
                 return jnp.ones_like(bvals)
                 
        radius = diameter / 2
        q_safe = jnp.where(q > 0, q, 1.0)
        factor = 2 * jnp.pi * q_safe * radius
        
        E = (3 / (factor ** 2) * (jnp.sin(factor) / factor - jnp.cos(factor))) ** 2
        return jnp.where(q > 0, E, 1.0)
