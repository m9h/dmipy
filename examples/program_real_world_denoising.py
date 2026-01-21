
"""
Testing Program Part 5: Real World Denoising (Global TV Regularization).

Demonstrates physics-based denoising on EDDEN data using GlobalAMICOSolver.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.inverse.global_amico import GlobalAMICOSolver

def run_denoising_demo():
    print("=== Testing Program: Real World Denoising (Global TV) ===")
    
    # 1. Load Data
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmarks/data/edden'))
    subject_dir = os.path.join(data_root, 'sub-01', 'ses-02', 'dwi')
    nii_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.nii.gz')
    bval_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.bval')
    bvec_path = os.path.join(subject_dir, 'sub-01_ses-02_dwi.bvec')
    
    if not os.path.exists(nii_path):
        print("Data not found.")
        return

    print("Loading NIfTI...")
    img = nib.load(nii_path)
    data = img.get_fdata()
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3: bvecs = bvecs.T
    
    # 2. Crop larger ROI for Denoising effect
    # 20x20 is good to see smoothing.
    cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    s = 20
    data_roi = data[cx-s:cx+s, cy-s:cy+s, cz:cz+1, :] # (40, 40, 1, N_meas)
    print(f"ROI Shape: {data_roi.shape}")
    
    scheme = JaxAcquisition(jnp.array(bvals), jnp.array(bvecs))
    
    # 3. Define Dictionary Model
    # Explicitly create dictionary of atoms.
    # 32 Sticks + 1 Ball.
    
    # Needs Wrapper with parameter_names for Solver
    class ModelWrapper:
        parameter_names = ('atom_idx',)
        
        def __init__(self, stick, ball, dirs):
            self.stick = stick
            self.ball = ball
            self.dirs_j = jnp.array(dirs)
            
        def __call__(self, bvals, gradient_directions, **kwargs):
            idx = kwargs['atom_idx']
            is_stick = idx < 32
            safe_idx = jnp.minimum(idx, 31)
            mu = self.dirs_j[safe_idx]
            
            s_stick = self.stick(bvals, gradient_directions, mu=mu)
            s_ball = self.ball(bvals, gradient_directions)
            return jnp.where(is_stick, s_stick, s_ball)

    stick = cylinder_models.C1Stick(lambda_par=1.7e-9)
    ball = gaussian_models.Ball(lambda_iso=3.0e-9)
    
    # Generate directions
    np.random.seed(42)
    dirs = np.random.randn(32, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    
    model = ModelWrapper(stick, ball, dirs)
    
    # 4. Initialize Solver
    print("Converting data to Dictionary representation...")
    N_atoms = 33
    solver = GlobalAMICOSolver(
        model=model,
        acquisition=scheme,
        dictionary_params={'atom_idx': jnp.arange(N_atoms)}
    )
    
    # 5. Fit Global with TV
    print("Running TV-Regularized Global Fit (lambda=0.5)...")
    # Using stronger lambda for visual effect demonstration
    coeffs_tv = solver.fit_global(data_roi, lambda_tv=0.5, lambda_l1=0.0)
    
    # 6. Fit Global without TV (Standard AMICO)
    print("Running Standard AMICO Fit (lambda=0.0)...")
    coeffs_no_tv = solver.fit_global(data_roi, lambda_tv=0.0, lambda_l1=0.0)
    
    # 7. Compare TV
    def compute_tv(img):
        diff_x = jnp.abs(img[:-1, :] - img[1:, :])
        diff_y = jnp.abs(img[:, :-1] - img[:, 1:])
        return jnp.sum(diff_x) + jnp.sum(diff_y)

    tv_cleaned = compute_tv(coeffs_tv)
    tv_noisy = compute_tv(coeffs_no_tv)
    
    print(f"Total Variation (Noisy Standard): {tv_noisy:.4f}")
    print(f"Total Variation (TV Cleaned): {tv_cleaned:.4f}")
    
    if tv_cleaned < tv_noisy:
        print("SUCCESS: TV Regularization significantly reduced spatial noise.")
    else:
        print("WARNING: Denoising effect not observed.")

if __name__ == "__main__":
    run_denoising_demo()
