import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import os
import time
from dmipy_jax.utils.spherical_harmonics import fit_spherical_harmonics
from dmipy_jax.acquisition import JaxAcquisition

def verify_pipeline():
    """
    Replicates the logic of procSTE.m:
    1. Load STE_degibbs_eddy.nii.gz
    2. Extract S0 and Diffusion data.
    3. Normalize S / mean(S0).
    4. Fit Spherical Harmonics (lmax=4, analogous to basisOrder=4 in procSTE.m).
    5. Save coefficients to NIfTI.
    """
    
    # 1. Paths
    # We use STE00_ExVivo as verified in previous steps
    base_path = os.path.expanduser("~/Downloads/STE/STE00_ExVivo/STE")
    nii_path = os.path.join(base_path, "STE_degibbs_eddy.nii.gz")
    bval_path = os.path.join(base_path, "bvals.txt")
    bvec_path = os.path.join(base_path, "bvecs.txt")
    
    print(f"Loading data from {base_path}...")
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine
    
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T
    if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
        bvecs = bvecs.T

    # 2. Prepare Data (procSTE.m: indB0 = bvals0==0)
    # Using small threshold for float bvals
    b0_mask = bvals < 50
    diff_mask = ~b0_mask
    
    S0 = np.mean(data[..., b0_mask], axis=-1)
    S = data[..., diff_mask]
    
    bvals_diff = bvals[diff_mask]
    bvecs_diff = bvecs[diff_mask]
    
    print(f"Data Shapes - S0: {S0.shape}, S: {S.shape}")
    print(f"Diffusion Volumes: {np.sum(diff_mask)}")
    
    # Normalize
    # Handle division by zero
    S0 = np.maximum(S0, 1e-6)
    S_normalized = S / S0[..., None]
    
    # 3. JAX Fit
    print("Starting JAX Spherical Harmonics Fit...")
    
    acq = JaxAcquisition(
        bvalues=jnp.array(bvals_diff),
        gradient_directions=jnp.array(bvecs_diff)
    )
    
    # procSTE.m uses basisOrder=4 for in vivo (CSA-ODF). 
    # For ex vivo it uses DTI. 
    # We will run SH fit order 4 for demonstration of the wrapper capability.
    lmax = 4
    
    # JIT Compile
    fit_fn = jax.jit(lambda s: fit_spherical_harmonics(s, acq, lmax=lmax))
    
    t0 = time.time()
    coeffs = fit_fn(jnp.array(S_normalized))
    # Block for timing
    coeffs.block_until_ready()
    duration = time.time() - t0
    
    print(f"Fit completed in {duration:.4f} seconds.")
    print(f"Coefficients shape: {coeffs.shape}")
    
    # 4. Save Results
    out_name = "STE_SH_Coeffs_lmax4.nii.gz"
    out_path = os.path.join(base_path, out_name)
    
    # Convert back to numpy
    coeffs_np = np.array(coeffs)
    
    new_img = nib.Nifti1Image(coeffs_np, affine)
    nib.save(new_img, out_path)
    
    print(f"Saved SH coefficients to {out_path}")
    print("Pipeline Verification COMPLETE.")

if __name__ == "__main__":
    verify_pipeline()
