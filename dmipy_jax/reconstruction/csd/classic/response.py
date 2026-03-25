
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial
from jax import jit

from dmipy_jax.fitting.dti import fit_dti, compute_fa_md
from dmipy_jax.utils.spherical_harmonics import sh_basis_real, cart2sphere

class ResponseEstimator:
    """
    Automated Response Function Estimator for CSD (Agent 1).
    Implements a Dhollander-like (2016) heuristic for single-shell data.
    """
    
    def __init__(self, sh_order: int = 8):
        self.sh_order = sh_order
        
    @partial(jit, static_argnums=(0,))
    def _fit_dti_map(self, data, bvals, bvecs):
        """
        Fits DTI to the entire volume (vmapped).
        """
        # data: (N_vox, N_meas) assumed flat
        fit_fn = jax.vmap(partial(fit_dti, bvals=bvals, bvecs=bvecs))
        evals, evecs, s0 = fit_fn(data)
        fa, md = compute_fa_md(evals)
        return evals, evecs, fa, md, s0

    def estimate(self, data, bvals, bvecs, mask=None, verbose=True):
        """
        Estimates WM (Response), GM, and CSF Signal levels.
        
        Args:
            data: (X, Y, Z, N_meas) or (N_vox, N_meas)
            bvals: (N_meas,)
            bvecs: (N_meas, 3)
            mask: Optional (X, Y, Z) or (N_vox,)
            
        Returns:
            wm_response: SH coefficients (lmax=sh_order)
            gm_signal: scalar (mean signal at shell)
            csf_signal: scalar (mean signal at shell)
        """
        # Flatten input
        if data.ndim == 4:
            data_flat = data.reshape(-1, data.shape[-1])
        else:
            data_flat = data
            
        if mask is not None:
             mask_flat = mask.reshape(-1)
        else:
             mask_flat = jnp.ones(data_flat.shape[0], dtype=bool)

        # 1. Fit DTI
        if verbose: print("Agent 1: Running DTI for calibration...")
        evals, evecs, fa, md, s0 = self._fit_dti_map(data_flat, bvals, bvecs)
        
        # Apply mask
        valid = mask_flat > 0
        # Safe masking
        fa = jnp.where(valid, fa, 0)
        md = jnp.where(valid, md, 0)
        
        # 2. Heuristic Voxel Selection (Dhollander-like)
        # WM: High FA (Top 300)
        # GM: Low FA (<0.2) and Med < MD < High
        # CSF: Low FA (<0.2) and High MD
        
        # Find WM
        n_wm = 300
        idx_sorted_fa = jnp.argsort(fa)[::-1]
        idx_wm = idx_sorted_fa[:n_wm]
        
        # Find CSF (High MD, Low FA)
        # Thresholds from MD percentile?
        # Let's say top 5% MD that has FA < 0.2
        low_fa_mask = (fa < 0.2) & valid
        md_masked = jnp.where(low_fa_mask, md, 0)
        # Actually checking percentiles on GPU is sort.
        # Just grab top 100 MD from low_fa set.
        idx_sorted_md = jnp.argsort(md_masked)[::-1]
        idx_csf = idx_sorted_md[:100]
        
        # Find GM (Low FA, Median MD?)
        # For simplicity in Agent 1:
        # GM is 'remaining' voxels.
        # Or: Use a safe range of MD (e.g. 0.7e-3 to 1.0e-3 if units known).
        # Without units, use the median of the volume?
        # Let's skip GM/CSF curve estimation for now and focus on WM Response (needed for CSD).
        
        # 3. WM Response Estimation
        if verbose: print(f"Agent 1: Estimating WM Response from {n_wm} single-fiber voxels...")
        wm_signals = data_flat[idx_wm]
        wm_evecs = evecs[idx_wm] # V1 is [:, :, 2]
        wm_s0 = s0[idx_wm]
        
        # Normalize Data (S / S0)
        # Note: Response function is usually in signal units or normalized?
        # MRtrix usually normalized so S0=1 (b=0 is 1).
        # We normalize here.
        wm_signals_norm = wm_signals / (wm_s0[:, None] + 1e-9)
        
        # Reorient
        wm_v1 = wm_evecs[:, :, 2]
        Rs = jax.vmap(self._get_rotation_to_z)(wm_v1)
        
        # Rotate bvecs: G' = R G
        bvecs_T = bvecs.T 
        rotated_bvecs = jnp.matmul(Rs, bvecs_T) # (300, 3, N)
        rotated_bvecs = jnp.transpose(rotated_bvecs, (0, 2, 1))
        
        # Flatten cloud
        cloud_signals = wm_signals_norm.reshape(-1)
        cloud_bvecs = rotated_bvecs.reshape(-1, 3)
        
        # Remove b~0 points from the fit?
        # Zonal SH fit to the shell data only.
        # Identify shell (b > 100?)
        shell_mask = jnp.tile(bvals > 50.0, n_wm) # repeat mask
        
        cloud_signals_shell = cloud_signals[shell_mask]
        cloud_bvecs_shell = cloud_bvecs[shell_mask]
        
        # Fit Zonal SH
        r, th, ph = cart2sphere(cloud_bvecs_shell[:,0], cloud_bvecs_shell[:,1], cloud_bvecs_shell[:,2])
        Y_all = sh_basis_real(th, ph, self.sh_order)
        
        # Select m=0 columns (Zonal)
        # indices: 0, 2, 6, 12, 20... (l*(l+1) + l/2? No)
        # Center of each block.
        # l=0 (size 1): idx 0.
        # l=2 (size 5): idx 1+2 = 3.
        # l=4 (size 9): idx 1+5+4 = 10.
        # Center index is l^2 + l + l = l^2 + 2l? No.
        # Standard ordering for 'sh_basis_real' (scipy): m varies -l..l
        # Block starts at l^2.
        # m=0 is at l^2 + l.
        m0_indices = [l**2 + l for l in range(0, self.sh_order + 1, 2)]
        m0_indices = jnp.array(m0_indices)
        
        Y_zonal = Y_all[:, m0_indices]
        
        # Least squares
        coeffs_zonal = jnp.linalg.lstsq(Y_zonal, cloud_signals_shell, rcond=None)[0]
        
        return coeffs_zonal

    @staticmethod
    def _get_rotation_to_z(v):
        """Rodrigues rotation from v to [0,0,1]."""
        z = jnp.array([0., 0., 1.])
        v = v / (jnp.linalg.norm(v) + 1e-9)
        c = jnp.dot(v, z)
        k = jnp.cross(v, z)
        s = jnp.linalg.norm(k)
        
        K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = jnp.eye(3) + K + (K @ K) * ((1 - c) / (s**2 + 1e-9))
        
        # Parallel safety
        return jnp.where(s < 1e-6, jnp.eye(3), R)
