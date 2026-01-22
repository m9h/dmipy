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
    parameter_names = ['mu', 'lambda_par', 'lambda_perp']
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'lambda_perp': 1}
    parameter_ranges = {
        'mu': ([-3.14, -6.28], [3.14, 6.28]), 
        'lambda_par': (0.1, 3.0), # um^2/ms
        'lambda_perp': (0.1, 3.0) 
    }
    parameter_scales = {'mu': 1.0, 'lambda_par': 1e-9, 'lambda_perp': 1e-9} # Scale for fitting?
    # Actually dmipy-jax usually handles scaling logic in Fitter.
    # Provided ranges should be reasonable.
    # If using SI in loop, ranges should be SI.
    # Our generated data was in SI (b ~ 3000 s/mm^2).
    # lambda ~ 1e-9 m^2/s = 1 um^2/ms.
    # Let's use SI in ranges: (1e-10, 3e-9)
    parameter_ranges = {
        'mu': ([-3.14, -6.28], [3.14, 6.28]), 
        'lambda_par': (0.0, 3e-9), 
        'lambda_perp': (0.0, 3e-9)
    }
    
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
