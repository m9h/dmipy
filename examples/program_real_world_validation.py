
"""
Testing Program Part 4: Real World Validation (EDDEN Dataset).

Validates dmipy-jax fitting on real-world multi-shell data (sub-01, ses-02).
Dataset: ds004910 (EDDEN).
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def load_edden_data(data_dir):
    # Paths
    subject_dir = os.path.join(data_dir, 'sub-01', 'ses-02', 'dwi')
    nii_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.nii.gz')
    bval_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.bval')
    bvec_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.bvec')
    
    # Load NIfTI
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # Load bvals/bvecs
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T # Usually (3, N) -> (N, 3)
    
    # Check shapes
    if bvecs.shape[1] != 3:
        # Sometimes (N, 3) in file?
        if bvecs.shape[0] == 3:
             bvecs = bvecs.T
    
    return data, bvals, bvecs, img.affine

def run_real_world_validation():
    print("=== Testing Program: Real World Validation (EDDEN) ===")
    
    # 1. Load Data
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmarks/data/edden'))
    if not os.path.exists(data_root):
        print(f"ERROR: Data directory not found: {data_root}")
        return

    print("Loading Data...")
    try:
        data, bvals, bvecs, affine = load_edden_data(data_root)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    print(f"Data Shape: {data.shape}")
    print(f"B-values: {bvals.shape}, Max b={bvals.max()}")
    
    # 2. Crop Data (Center Slice)
    # Full fit might take too long for interactive test.
    # Data is usually (X, Y, Z, N_meas)
    cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    
    # Take a 4x4 patch in center slice
    s = 2
    data_roi = data[cx-s:cx+s, cy-s:cy+s, cz:cz+1, :]
    print(f"ROI Shape: {data_roi.shape}")
    
    # 3. Setup Acquisition
    # EDDEN usually has variable TE/TR? Assuming single shell block for now or just passing bvals/bvecs.
    # For simple C1Stick+Ball, we need bvals/bvecs. 
    # Timing (delta/Delta) is needed for RestrictedCylinder but BallStick is often Gaussian regime approx.
    # We'll treat C1 as Zeppelins/Sticks (Gaussian).
    
    scheme = JaxAcquisition(
        bvalues=jnp.array(bvals),
        gradient_directions=jnp.array(bvecs)
    )
    
    # 4. Define Model: Ball + Stick
    # Common fit for dMRI.
    # Stick: Axial diffusion (intra-axonal)
    # Ball: Isotropic diffusion (extra-axonal / CSF)
    # We optimize parameters: Stick orientation (mu), Stick diffusivity? (often fixed), Ball diffusivity? (often fixed or fit), Partial Volume (f).
    
    # Let's fit:
    # 1. C1Stick (fit mu, fix lambda_par=1.7e-9 for stability or fit it?)
    # 2. Ball (fit lambda_iso? or fix to 3e-9)
    # Simplest: WatsonNODDI style -> Stick (fixed diff) + Ball (fixed diff)?
    # Let's try fitting lambda to see optimizer power.
    
    stick = cylinder_models.C1Stick() # Fits mu, lambda_par
    ball = gaussian_models.Ball() # Fits lambda_iso
    
    model = JaxMultiCompartmentModel([stick, ball])
    
    # Ranges? 
    # C1Stick lambda_par: (0.1, 3.0) -> set in class
    # Ball lambda_iso: (0, 3.0) -> set in class
    
    print("Fitting Model: C1Stick + Ball using LBFGSB (Bounded)...")
    
    # 5. Fit
    # data_roi is (20, 20, 1, N_meas) matches our updated fit() flattened logic.
    fitted = model.fit(scheme, jnp.array(data_roi), method="LBFGSB")
    
    # 6. Analyze Results
    print("Fitting Complete.")
    
    # Inspect fitted parameters
    # stick_mu, stick_lambda_par, ball_lambda_iso, partial_volumes
    
    # Check shape of output
    # fitted is dictionary
    for k, v in fitted.items():
        print(f"Param '{k}' shape: {v.shape}")
        
    # Simple metric: Compute FA from Stick (if purely stick, FA=1). 
    # Actually, let's just show mean lambda_par
    
    l_par = fitted.get('lambda_par', fitted.get('C1Stick_1_lambda_par'))
    if l_par is not None:
        l_par_mean = jnp.mean(l_par)
        print(f"Mean ROI Stick Diffusivity: {l_par_mean:.4e} m^2/s")
        
    # Check for NaNs
    n_nans = jnp.sum(jnp.isnan(l_par)) if l_par is not None else 0
    if n_nans > 0:
        print(f"WARNING: {n_nans} NaNs detected in fit.")
    else:
        print("SUCCESS: No NaNs in fit.")

if __name__ == "__main__":
    run_real_world_validation()
