
"""
Program: 7T Voxel-wise Interaction Demo
Objective: Demonstrate that dmipy-jax fits physically consistent models across spatially coupled voxels, exploiting the 800µm resolution.

This script:
1. Generates synthetic 7T data (0.8mm isotropic) with a sharp boundary feature (White Matter / Gray Matter interface).
2. Fits a Microstructure Model (C1Stick + Zeppelin) using:
    a. Voxel-wise Independent Fitting (Standard)
    b. Spatially Coupled Fitting (TV Regularization)
3. Compares the reconstruction quality, highlighting the preservation of sharp boundaries and noise suppression.

Note: While the prompt suggested ds003216/ds003563, due to S3 access limitations, we prioritize a robust synthetic 7T demonstration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
import sys
import os

# Ensure project modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dmipy_jax.signal_models import cylinder_models
from dmipy_jax.signal_models import gaussian_models
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.inverse.global_amico import GlobalAMICOSolver

# --- 1. Define Models ---

class Zeppelin(eqx.Module):
    """
    Zeppelin model (Cylindrically Symmetric Tensor).
    """
    mu: jnp.ndarray = None
    lambda_par: float = None
    lambda_perp: float = None
    
    parameter_names = ('mu', 'lambda_par', 'lambda_perp')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'lambda_perp': 1}
    # Ranges are illustrative
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'lambda_perp': (0.1e-9, 3e-9)
    }

    def __init__(self, mu=None, lambda_par=None, lambda_perp=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.lambda_perp = lambda_perp

    def __call__(self, bvals, gradient_directions, **kwargs):
        l_par = kwargs.get('lambda_par', self.lambda_par)
        l_perp = kwargs.get('lambda_perp', self.lambda_perp)
        mu = kwargs.get('mu', self.mu)
        
        # Convert params to array if needed
        # Assume mu is cartesian in this simple demo context or mapped externally
        # The solver passes 'mu' from logical dictionary.
        
        # Ensure mu is cartesian (3,)
        mu = jnp.asarray(mu)
        if mu.size == 2:
             # Spherical to Cartesian
             theta, phi = mu[0], mu[1]
             st, ct = jnp.sin(theta), jnp.cos(theta)
             sp, cp = jnp.sin(phi), jnp.cos(phi)
             mu = jnp.array([st*cp, st*sp, ct])
        
        return gaussian_models.g2_zeppelin(bvals, gradient_directions, mu, l_par, l_perp)

# --- 2. Real 7T Data Loading ---

def load_real_7T_data(base_path='data/ds003563'):
    """
    Loads and concatenates 4 sessions of sub-yv98 (ses-4291, 4297, 4284, 4295).
    """
    import nibabel as nib
    import numpy as np
    
    sessions = ['ses-4291', 'ses-4295', 'ses-4284', 'ses-4297']
    
    data_list = []
    bvals_list = []
    bvecs_list = []
    
    print(f"Loading {len(sessions)} sessions from {base_path}...")
    
    for ses in sessions:
        dwi_path = os.path.join(base_path, f'sub-yv98/{ses}/dwi/sub-yv98_{ses}_dwi.nii.gz')
        bval_path = os.path.join(base_path, f'sub-yv98/{ses}/dwi/sub-yv98_{ses}_dwi.bval')
        bvec_path = os.path.join(base_path, f'sub-yv98/{ses}/dwi/sub-yv98_{ses}_dwi.bvec')
        
        # Load NIfTI
        img = nib.load(dwi_path)
        data = img.get_fdata() # (X, Y, Z, T)
        data_list.append(data)
        
        # Load bvals/bvecs
        bvals = np.loadtxt(bval_path) # (T,)
        bvecs = np.loadtxt(bvec_path) # (3, T) usually
        
        # Ensure shapes
        if bvecs.shape[0] == 3:
            bvecs = bvecs.T # Make (T, 3)
            
        bvals_list.append(bvals)
        bvecs_list.append(bvecs)
        
    # Concatenate
    full_data = np.concatenate(data_list, axis=-1)
    full_bvals = np.concatenate(bvals_list, axis=0)
    full_bvecs = np.concatenate(bvecs_list, axis=0)
    
    print(f"Combined Data Shape: {full_data.shape}")
    print(f"Combined Bvals Shape: {full_bvals.shape}")
    
    # Create Acquisition Object
    class Acquisition:
        bvalues = jnp.array(full_bvals)
        gradient_directions = jnp.array(full_bvecs)
        # Approximate timings for 7T (often long delta/Delta)
        delta = 0.03
        Delta = 0.04
        
    return full_data, Acquisition()

# --- 3. Main Routine ---

def run_demo():
    print("=== 7T Voxel-wise vs Coupled Fit Demo (OpenNeuro ds003563) ===")
    
    # 1. Load Real Data
    try:
        data_full, scheme = load_real_7T_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Please ensure you have run 'datalad get' for sessions 4291, 4295, 4284, 4297")
        return

    # 2. Crop Region (Focus on brain center)
    # Original: (240, 240, 35)
    # Center crop 20x20x10?
    # Let's verify where the brain is. 35 slices is thin.
    # Center X, Y is 120.
    cx, cy, cz = 120, 120, 17
    wx, wy, wz = 20, 20, 5 # Half-widths
    # Full crop 40x40x10
    
    # Let's take 20x20x1 (Single slice for clarity in default demo) or small block
    # Instructions: "Crop a small 20x20x20 region"
    # But Z is only 35. So 20 is fine.
    
    s_x = slice(cx-10, cx+10)
    s_y = slice(cy-10, cy+10)
    s_z = slice(cz-10, cz+10)
    
    data_crop = jnp.array(data_full[s_x, s_y, s_z, :])
    print(f"Cropped Data Shape: {data_crop.shape}")
    
    # Normalize data (AMICO expects usually normalized or raw? 
    # GlobalAMICO has data fidelity. 
    # Better to normalize by mean b0)
    
    b0_mask = scheme.bvalues < 50
    b0_mean = jnp.mean(data_crop[..., b0_mask], axis=-1, keepdims=True)
    b0_mean = jnp.maximum(b0_mean, 1.0) # Avoid div zero
    
    data_norm = data_crop / b0_mean
    
    # 3. Dictionary Setup (C1Stick + Zeppelin/Ball)
    # b=750 is low. We use Stick + Ball to emulate "directions + isotropic".
    # Zeppelin with perp=0 is Stick.
    # Zeppelin with perp=par is Ball.
    
    print("Building Dictionary...")
    # Generate directions on Sphere (Fibonacci)
    def fibonacci_sphere(samples=1):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    N_atoms = 32
    dirs = fibonacci_sphere(N_atoms)
    
    # Dictionary Model: Sticks + 1 Ball
    stick = cylinder_models.C1Stick(lambda_par=2.0e-9)
    # Ball (Zeppelin isotropic)
    ball = gaussian_models.Ball(lambda_iso=0.7e-9) # GM diffusivity approx
    
    class DictionaryModel:
        parameter_names = ('atom_idx',)
        def __init__(self, stick, ball, dirs):
            self.stick = stick
            self.ball = ball
            self.dirs = jnp.array(dirs)
            
        def __call__(self, bvals, gradient_directions, **kwargs):
            idx = kwargs['atom_idx']
            # If idx < N_atoms: Stick
            # If idx == N_atoms: Ball
            
            is_stick = idx < len(self.dirs)
            safe_idx = jnp.minimum(idx, len(self.dirs)-1)
            mu = self.dirs[safe_idx]
            
            s_stick = self.stick(bvals, gradient_directions, mu=mu)
            s_ball = self.ball(bvals, gradient_directions)
            
            return jnp.where(is_stick, s_stick, s_ball)
            
    dict_model = DictionaryModel(stick, ball, dirs)
    
    # Dictionary Params: 0..31 (Sticks), 32 (Ball)
    dict_params = {'atom_idx': jnp.arange(N_atoms + 1)}
    
    # 4. Solvers
    solver = GlobalAMICOSolver(
        model=dict_model,
        acquisition=scheme,
        dictionary_params=dict_params
    )
    
    # 5. Fit Voxel-wise (TV=0)
    print("Fitting Voxel-wise (TV=0)...")
    coeffs_vx = solver.fit_global(data_norm, lambda_tv=0.0, maxiter=30, lambda_l1=0.001)
    
    # 6. Fit Coupled (TV=0.1)
    # Using smaller TV as real data might be less "blocky" than synthetic
    print("Fitting Coupled (TV=0.1)...")
    coeffs_tv = solver.fit_global(data_norm, lambda_tv=0.1, maxiter=30, lambda_l1=0.001)
    
    # 7. Analysis - Focus on center slice
    # Show Volume Fraction of Sticks (Anisotropy Map proxy)
    # Index 0..31 are sticks. 32 is Ball.
    # VF_Stick = sum(coeffs[..., 0:32])
    
    vf_vx = jnp.sum(coeffs_vx[..., :N_atoms], axis=-1)
    vf_tv = jnp.sum(coeffs_tv[..., :N_atoms], axis=-1)
    
    # Take center slice for viz
    mid_z = vf_vx.shape[2] // 2
    map_vx = vf_vx[:, :, mid_z]
    map_tv = vf_tv[:, :, mid_z]
    
    # Calculate metrics
    # Simple Gradient Norm
    g_vx = jnp.gradient(map_vx)
    grad_vx = jnp.mean(jnp.sqrt(g_vx[0]**2 + g_vx[1]**2))
    
    g_tv = jnp.gradient(map_tv)
    grad_tv = jnp.mean(jnp.sqrt(g_tv[0]**2 + g_tv[1]**2))
    
    print(f"\nResults (Mean Gradient Magnitude):")
    print(f"  Voxel-wise: {float(grad_vx):.6f}")
    print(f"  Coupled:    {float(grad_tv):.6f}")
    
    # ASCII Viz
    def ascii_heatmap(img):
        chars = " .:-=+*#%@"
        if img.max() > img.min():
            norm_img = (img - img.min()) / (img.max() - img.min())
        else:
            norm_img = img
        for row in norm_img:
            line = ""
            for val in row:
                idx = int(np.clip(val * 9, 0, 9))
                line += chars[idx]
            print(line)
            
    print("\nVoxel-wise Map (Center Slice):")
    ascii_heatmap(np.array(map_vx))
    
    print("\nCoupled Map (Center Slice):")
    ascii_heatmap(np.array(map_tv))
    
    print("\n[Comparison Complete]")

if __name__ == "__main__":
    run_demo()
