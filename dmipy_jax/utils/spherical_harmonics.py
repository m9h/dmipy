import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm

def cart2sphere(x, y, z):
    """
    Convert Cartesian coordinates to Spherical coordinates.
    Returns (r, theta, phi).
    theta: inclination [0, pi]
    phi: azimuth [-pi, pi]
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.where(r > 0, jnp.arccos(jnp.clip(z / r, -1.0, 1.0)), 0.0)
    phi = jnp.arctan2(y, x)
    return r, theta, phi

def sh_basis_real(theta, phi, lmax):
    """
    Computes Real Spherical Harmonics Basis Y_lm(theta, phi).
    
    Order: l=0, l=2, ... up to lmax (even only).
    Within l: m = -l ... l.
    """
    coeffs_list = []
    
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            y_c = sph_harm(m, l, phi, theta) # Note arguments (m, n, theta=azimuth, phi=polar) in some libs, but cart2sphere aligns with physics.
            # Scipy sph_harm(m, n, theta, phi): theta=azimuth, phi=polar.
            # My cart2sphere: theta=polar, phi=azimuth.
            # So pass (m, l, phi, theta).
            
            if m < 0:
                y_r = jnp.sqrt(2) * ((-1)**m) * jnp.imag(y_c)
            elif m == 0:
                y_r = jnp.real(y_c)
            else: # m > 0
                y_r = jnp.sqrt(2) * ((-1)**m) * jnp.real(y_c)
            
            coeffs_list.append(y_r)
            
    return jnp.stack(coeffs_list, axis=-1)
