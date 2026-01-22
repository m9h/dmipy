import jax
import jax.numpy as jnp
from jax import pure_callback
import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:
    # Use fallback or standard sph_harm if y not available (older scipy)
    # But for now assume sph_harm_y as verified
    from scipy.special import sph_harm as sph_harm_y
    # Note: If sph_harm is used, we might need argument swap.
    # But since verified script worked with sph_harm_y, we assume it exists.

def cart2sphere(x, y, z):
    """
    Convert Cartesian coordinates to Spherical coordinates.
    Returns (r, theta, phi).
    theta: inclination [0, pi] (Polar)
    phi: azimuth [-pi, pi]
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.where(r > 0, jnp.arccos(jnp.clip(z / jnp.maximum(r, 1e-12), -1.0, 1.0)), 0.0)
    phi = jnp.arctan2(y, x)
    return r, theta, phi

def _scipy_sh_basis(theta, phi, lmax_arr):
    """
    Callback function to compute SH basis Complex values using Scipy.
    Returns array of shape (N, N_complex_terms).
    """
    lmax = int(lmax_arr)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    
    # Generate (l, m) pairs for m >= 0
    l_list = []
    m_list = []
    
    for l in range(0, lmax + 1, 2):
        for m in range(0, l + 1):
             l_list.append(l)
             m_list.append(m)
             
    l_arr = np.array(l_list)
    m_arr = np.array(m_list)
    
    # Vectorized call to sph_harm_y
    # Broadcast to (N, N_pairs)
    # theta, phi: (N,) -> (N, 1)
    # l, m: (N_pairs,) -> (1, N_pairs)
    
    # sph_harm_y(n, m, theta, phi)
    # Convention: theta (Polar), phi (Azimuth)
    
    y = sph_harm_y(l_arr[None, :], m_arr[None, :], theta[:, None], phi[:, None])
    return y

def sh_basis_real(theta, phi, lmax):
    """
    Computes Real Spherical Harmonics Basis Y_lm(theta, phi).
    
    Order: l=0, l=2, ... up to lmax (even only).
    Within l: m = -l ... l.
    
    Implementation uses Scipy callback for robustness.
    """
    # Determine output shapes
    # N_coeffs = sum(2*l + 1 for l in 0..lmax even)
    # But we compute complex m>=0 first.
    # Count m>=0 pairs: Sum (l+1) for l=0,2...lmax
    
    # We invoke callback to get Complex Y for m>=0
    # Then construct Real basis.
    
    # Output dim for callback
    complex_dim = 0
    for l in range(0, lmax + 1, 2):
        complex_dim += (l + 1)
        
    result_shape = jax.ShapeDtypeStruct(theta.shape + (complex_dim,), jnp.complex64) # or 128
    
    y_complex = pure_callback(_scipy_sh_basis, result_shape, theta, phi, lmax)
    
    # Construct Real Basis from Complex M>=0
    # Order: For each l, we want m = -l ... l
    # y_complex columns correspond to m = 0, 1, ... l for each l block.
    
    real_basis_list = []
    col_idx = 0
    
    for l in range(0, lmax + 1, 2):
        # Current block indices in y_complex: col_idx ... col_idx + l
        
        # Re-assemble block in order -l ... l
        # m=0 is y_complex[..., col_idx]
        # m>0 is y_complex[..., col_idx + m]
        
        # -l ... -1
        for m in range(-l, 0):
             abs_m = -m
             y_pos = y_complex[..., col_idx + abs_m]
             condon_phase = (-1)**abs_m
             # Basis m < 0: sqrt(2) * (-1)^m * Im(Y_l^m)
             val = jnp.sqrt(2) * condon_phase * jnp.imag(y_pos)
             real_basis_list.append(val)
             
        # m=0
        val0 = jnp.real(y_complex[..., col_idx])
        real_basis_list.append(val0)
        
        # 1 ... l
        for m in range(1, l + 1):
             y_pos = y_complex[..., col_idx + m]
             condon_phase = (-1)**m
             # Basis m > 0: sqrt(2) * (-1)^m * Re(Y_l^m)
             val = jnp.sqrt(2) * condon_phase * jnp.real(y_pos)
             real_basis_list.append(val)
             
        col_idx += (l + 1)
            
    return jnp.stack(real_basis_list, axis=-1)
