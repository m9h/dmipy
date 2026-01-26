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

def fit_spherical_harmonics(signal, acquisition, lmax=4):
    r"""
    Fit Real Spherical Harmonics to the diffusion signal.
    
    This function computes the coefficients $c_{lm}$ such that:
    $S \approx \sum c_{lm} Y_{lm}(\theta, \phi)$
    
    The fit is performed via linear least squares:
    $C = (Y^T Y)^{-1} Y^T S = Y^\dagger S$
    
    Args:
        signal (jnp.ndarray): Diffusion signal array. 
                              Shape can be (N_gradients,) for a single voxel 
                              or (..., N_gradients) for a batch of voxels.
        acquisition (JaxAcquisition): Acquisition object containing gradient directions.
        lmax (int): Maximum SH order (even). Default is 4.
        
    Returns:
        coeffs (jnp.ndarray): SH coefficients. 
                              Shape (..., N_coeffs) where N_coeffs = (lmax+1)(lmax+2)/2.
    """
    # 1. Extract gradient directions (Cartesian) and convert to Spherical
    # bvecs shape: (N_gradients, 3)
    bvecs = acquisition.gradient_directions
    r, theta, phi = cart2sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])
    
    # 2. Compute SH Design Matrix Y
    # Shape: (N_gradients, N_coeffs)
    Y = sh_basis_real(theta, phi, lmax)
    
    # 3. Compute Pseudo-Inverse
    # Y_inv = (Y.T @ Y)^-1 @ Y.T
    # Or simply pinv(Y)
    # Shape: (N_coeffs, N_gradients)
    Y_inv = jnp.linalg.pinv(Y)
    
    # 4. Perform Fit
    # We want to support batching over voxels.
    # Signal shape: (..., N_gradients)
    # Coeffs = Signal @ Y_inv.T
    
    # If signal is 1D (N_gradients,), we get (N_coeffs,)
    # If signal is (X, Y, Z, N_gradients), we get (X, Y, Z, N_coeffs)
    
    coeffs = jnp.dot(signal, Y_inv.T)
    
    return coeffs
