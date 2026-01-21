import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any
from dmipy_jax.signal_models import g2_zeppelin

class Zeppelin(eqx.Module):
    r"""
    The Zeppelin model [1]_ - an axially symmetric Tensor.
    
    Args:
        mu: (3,) Fiber orientation.
        lambda_par: Diffusivity along the axis.
        lambda_perp: Diffusivity perpendicular to the axis.
    """
    
    mu: Optional[Any] = None
    lambda_par: Optional[Any] = None
    lambda_perp: Optional[Any] = None

    def __call__(self, bvals, gradient_directions, **kwargs):
        # Resolve parameters: override if provided in kwargs, else use stored class attributes
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        lambda_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)

        # Convert spherical [theta, phi] to cartesian vector if needed
        # Assuming input 'mu' can be spherical (2,) or cartesian (3,)
        # The original code assumed spherical input [theta, phi] and converted to cartesian.
        
        # Check matching standard dmipy behavior: usually params are provided as kwargs during fitting
        # If mu has shape (2,), treat as spherical.
        
        mu = jnp.array(mu)
        if mu.shape[-1] == 2:
            theta = mu[..., 0]
            phi = mu[..., 1]
            st = jnp.sin(theta)
            ct = jnp.cos(theta)
            sp = jnp.sin(phi)
            cp = jnp.cos(phi)
            mu_cart = jnp.stack([st * cp, st * sp, ct], axis=-1)
        else:
            mu_cart = mu

        return g2_zeppelin(bvals, gradient_directions, mu_cart, lambda_par, lambda_perp)
