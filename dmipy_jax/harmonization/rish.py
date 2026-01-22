import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple
from dmipy_jax.utils.spherical_harmonics import sh_basis_real

def fit_sh_coeffs(signal: jnp.ndarray, bvecs: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """
    Fits Real Spherical Harmonics coefficients to the signal.
    
    Args:
        signal: (N,) or (..., N) array of dMRI signals.
        bvecs: (N, 3) array of gradient directions.
        lmax: Maximum harmonic order (must be even).
        
    Returns:
        coeffs: (..., N_coeffs) array of SH coefficients.
    """
    # Convert bvecs to spherical coordinates (r, theta, phi)
    # Note: sh_basis_real expects (theta, phi) where theta is polar, phi is azimuth
    x, y, z = bvecs.T
    # dmipy_jax.utils.spherical_harmonics.cart2sphere returns (r, theta, phi)
    # but we can just do it inline here to be sure or import it if readily available.
    # Let's trust the input bvecs are normalized.
    
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    theta = jnp.where(r > 0, jnp.arccos(jnp.clip(z / jnp.maximum(r, 1e-12), -1.0, 1.0)), 0.0)
    phi = jnp.arctan2(y, x)
    
    # Compute Basis Y: (N, N_coeffs)
    Y = sh_basis_real(theta, phi, lmax)
    
    # Compute Basis Y: (N, N_coeffs)
    Y = sh_basis_real(theta, phi, lmax)
    
    # Flatten signal batch dimensions
    input_shape = signal.shape
    n_dirs = input_shape[-1]
    
    # Check dimensions
    if n_dirs != Y.shape[0]:
        raise ValueError(f"Signal last dimension ({n_dirs}) must match Y rows ({Y.shape[0]})")

    batch_shape = input_shape[:-1]
    n_coeffs = Y.shape[1]
    
    signal_flat = signal.reshape(-1, n_dirs)
    
    # Use vmap to solve for each voxel/sample
    # This avoids broadcasting Y and potential solver issues with 3D inputs
    def solve_one(s):
        # rcond=1e-5 for stability
        return jnp.linalg.lstsq(Y, s, rcond=None)[0]
    
    coeffs_flat = jax.vmap(solve_one)(signal_flat)
    
    return coeffs_flat.reshape(batch_shape + (n_coeffs,))

def compute_rish_features(coeffs: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """
    Computes Rotation Invariant Spherical Harmonics (RISH) features.
    
    Args:
        coeffs: (..., N_coeffs) SH coefficients.
        lmax: Maximum harmonic order.
        
    Returns:
        rish: (..., N_orders) RISH features (L2 energy per order).
              Orders correspond to l=0, 2, ..., lmax.
    """
    rish_list = []
    start_idx = 0
    for l in range(0, lmax + 1, 2):
        n_m = 2 * l + 1
        end_idx = start_idx + n_m
        
        c_l = coeffs[..., start_idx:end_idx]
        
        # Energy = sqrt( sum( |c_lm|^2 ) )
        energy = jnp.linalg.norm(c_l, axis=-1)
        rish_list.append(energy)
        
        start_idx = end_idx
        
    return jnp.stack(rish_list, axis=-1)

class RISHHarmonizer(eqx.Module):
    """
    Harmonizes dMRI data using RISH features to learn a scale map between sites.
    Reference: Mirzaalian et al., 2015.
    """
    scale_factors: jnp.ndarray
    lmax: int

    def __init__(self, lmax: int = 4, scale_factors: Optional[jnp.ndarray] = None):
        self.lmax = lmax
        if scale_factors is None:
            # Initialize with ones (no scaling). 
            # Number of even orders up to lmax: l=0, 2, ..., lmax -> lmax//2 + 1
            self.scale_factors = jnp.ones(lmax // 2 + 1)
        else:
            self.scale_factors = scale_factors

    def fit(self, 
            ref_signals: jnp.ndarray, 
            ref_bvecs: jnp.ndarray, 
            tgt_signals: jnp.ndarray, 
            tgt_bvecs: jnp.ndarray,
            epsilon: float = 1e-6):
        """
        Learns scale factors from Reference and Target populations.
        
        Args:
            ref_signals: (Batch_Ref, N_dirs_ref) Signals from Reference site.
            ref_bvecs: (N_dirs_ref, 3) 
            tgt_signals: (Batch_Tgt, N_dirs_tgt) Signals from Target site.
            tgt_bvecs: (N_dirs_tgt, 3)
            
        Returns:
            A new RISHHarmonizer instance with learned scale factors.
        """
        # Fit SH to Reference
        c_ref = fit_sh_coeffs(ref_signals, ref_bvecs, self.lmax)
        r_ref = compute_rish_features(c_ref, self.lmax) # (Batch_Ref, N_orders)
        
        # Fit SH to Target
        c_tgt = fit_sh_coeffs(tgt_signals, tgt_bvecs, self.lmax)
        r_tgt = compute_rish_features(c_tgt, self.lmax) # (Batch_Tgt, N_orders)
        
        # Compute Means (Scalar for global harmonization)
        # TODO: Support voxel-wise maps if input has spatial dims.
        # Assuming input is just a batch of voxels from unmatched but similar populations.
        mean_ref = jnp.mean(r_ref, axis=0) # (N_orders,)
        mean_tgt = jnp.mean(r_tgt, axis=0) # (N_orders,)
        
        # Compute Scale Map
        # S_l = R_ref_l / R_tgt_l
        new_scale_factors = mean_ref / (mean_tgt + epsilon)
        
        return eqx.tree_at(lambda x: x.scale_factors, self, new_scale_factors)

    def __call__(self, signal: jnp.ndarray, bvecs: jnp.ndarray) -> jnp.ndarray:
        """
        Harmonizes the input signal (Target -> Reference space).
        
        Args:
            signal: (..., N_dirs) Input signal from Target site.
            bvecs: (N_dirs, 3) Gradient directions.
            
        Returns:
            harmonized_signal: (..., N_dirs)
        """
        # 1. Fit SH
        coeffs = fit_sh_coeffs(signal, bvecs, self.lmax)
        
        # 2. Apply Scale Factors to Coefficients
        # We need to broadcast scale factors to full coefficient vector
        # scale_factors: (N_orders,) -> e.g. [s0, s2, s4]
        # coeffs: [c00, c2-2...c22, c4-4...c44]
        
        # Expand scale factors
        expanded_scales = []
        for i, l in enumerate(range(0, self.lmax + 1, 2)):
            n_m = 2 * l + 1
            s_l = self.scale_factors[i]
            expanded_scales.append(jnp.full((n_m,), s_l))
            
        full_scale_vector = jnp.concatenate(expanded_scales)
        
        # Apply scaling
        coeffs_prime = coeffs * full_scale_vector
        
        # 3. Reconstruct Signal
        # S' = Y * C'
        # We need Y for the *input* bvecs (harmonized signal is on same gradients)
        x, y, z = bvecs.T
        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.where(r > 0, jnp.arccos(jnp.clip(z / jnp.maximum(r, 1e-12), -1.0, 1.0)), 0.0)
        phi = jnp.arctan2(y, x)
        
        Y = sh_basis_real(theta, phi, self.lmax)
        
        # coeffs_prime: (..., N_coeffs)
        # Y: (N_dirs, N_coeffs)
        # Result: (..., N_dirs)
        
        harmonized_signal = jnp.dot(coeffs_prime, Y.T)
        
        return harmonized_signal
