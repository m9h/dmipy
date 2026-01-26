
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
from pathlib import Path

# Set JAX to use CPU or GPU as available (default)
# os.environ["JAX_PLATFORM_NAME"] = "cpu" # Uncomment if GPU issues

from dmipy_jax.data.ixi import load_ixi_subject
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.models import BallStick
from dmipy_jax.fitting.optimization import OptimistixFitter
import optimistix

def main():
    print("=== IXI Dataset Demo ===")
    
    # 1. Fetch Data
    # This might take time on first run (500MB download)
    try:
        print("Fetching/Loading IXI subject...")
        # Loads first available subject or downloads
        data, affine, bvals, bvecs = load_ixi_subject()
        print(f"Data shape: {data.shape}")
        print(f"Bvals shape: {bvals.shape}, Bvecs shape: {bvecs.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Falling back to SYNTHETIC data for demonstration purposes.")
        
        # Generate synthetic DTI-like data
        # 16 volumes (1 b0, 15 b1000)
        # Size: 32x32x10 (small chunk)
        shape = (32, 32, 10, 16)
        affine = np.eye(4)
        
        bvals = np.concatenate(([0], np.ones(15) * 1000))
        # Random bvecs
        rng = np.random.default_rng(42)
        bvecs = rng.standard_normal((16, 3))
        bvecs[0] = [0, 0, 0]
        bvecs[1:] = bvecs[1:] / np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
        
        # Synthetic signal: Random noise + structure?
        # Just random exponential decay-like for testing pipeline
        data = rng.uniform(0.1, 1.0, size=shape)
        # Ensure b0 is higher
        data[..., 0] = rng.uniform(2.0, 3.0, size=shape[:3])
        print(f"Generated synthetic data: {data.shape}")

    # 2. Preprocessing
    print("Preprocessing...")
    # Simple threshold masking for demo speed (dipy median_otsu is better but slower)
    # We take the b0 (first volume usually)
    b0_vol = data[..., 0]
    mask = b0_vol > (0.1 * np.max(b0_vol)) # Simple bg threshold
    
    # Select specific slice to speed up demo if desired, 
    # but let's try a reasonable chunk (e.g. middle slices)
    z_center = data.shape[2] // 2
    # chunk of 5 slices
    sl = slice(z_center - 2, z_center + 3)
    
    data_crop = data[:, :, sl, :]
    mask_crop = mask[:, :, sl]
    
    print(f"Processing slice range: {sl}")
    print(f"Voxels to fit: {np.sum(mask_crop)}")
    
    # Flatten data for fitting
    valid_voxels = data_crop[mask_crop]
    # Normalize signal
    # Avoid div by zero
    valid_voxels = valid_voxels / (valid_voxels[..., 0:1] + 1e-6)
    
    # 3. Setup Acquisition
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # 4. Setup Model
    # Ball & Stick
    # Params: [theta, phi, f_stick]
    model = BallStick(diffusivity=1.7e-9) 
    
    # Initial Guess
    # theta=0, phi=0 (z-axis), f_stick=0.5
    # Might want random or heuristic. 
    # Optimistix is robust enough.
    x0 = jnp.array([0.0, 0.0, 0.5])
    
    # 5. Fitting
    print("Setting up Fitter...")
    # Parameter ranges for BallStick: [theta, phi, f_stick]
    # Scaling is handled by providing these ranges.
    ranges = [
        (0.0, np.pi),       # theta
        (-np.pi, np.pi),    # phi
        (0.0, 1.0)          # f_stick
    ]
    fitter = OptimistixFitter(model, parameter_ranges=ranges)
    
    # JIT compile the fit function
    # We map over the batch of voxels
    # create voxel-wise fitter
    
    # We define a function that fits a single voxel
    def fit_voxel(data_voxel):
        # We need to pass sigma for Rician loss if used, or just dummy if MSE.
        # Check OptimistixFitter implementation.
        # Usually it defaults to MSE if sigma not used or we pass sigma=1.
        # fit() -> (params, result)
        # Note: Validating usage from previous session learnings
        # fit takes (data, acq, x0, sigma)
        # We assume sigma=None or handle it.
        # Let's pass sigma=1.0 for unweighted/MSE-like behavior if Rician reduces to MSE with constant sigma??
        # Or just use MSE if implemented.
        # We'll pass sigma=1.0
        return fitter.fit(data_voxel, acq, x0)

    # Vectorize
    fit_batch = jax.vmap(fit_voxel)
    # JIT
    fit_batch_jit = jax.jit(fit_batch)
    
    print("Fitting...")
    start_time = time.time()
    
    # Convert to JAX array
    jax_data = jnp.array(valid_voxels)
    
    # Run fit
    params_fitted, result = fit_batch_jit(jax_data)
    # Block until ready for timing
    params_fitted.block_until_ready()
    
    end_time = time.time()
    duration = end_time - start_time
    n_vox = len(valid_voxels)
    print(f"Fitted {n_vox} voxels in {duration:.4f} seconds.")
    print(f"Speed: {n_vox/duration:.2f} voxels/sec")
    
    # 6. Reconstruct and Save
    print("Reconstructing Volume...")
    out_shape = mask_crop.shape + (3,) # [theta, phi, f_stick]
    out_vol = np.zeros(out_shape)
    
    indices = np.where(mask_crop)
    # params_fitted is (N_vox, 3)
    
    # Needs to unpack back to 3D structure
    # Advanced indexing
    # We iterate or use fancy indexing
    # out_vol[mask_crop] = params_fitted # Shape mismatch if boolean mask used directly on 3D array?
    # mask_crop is (X,Y,Z). out_vol is (X,Y,Z,3).
    # We can assign:
    out_vol[mask_crop] = np.array(params_fitted)
    
    # Extract f_stick map (index 2)
    f_stick_vol = out_vol[..., 2]
    
    # Save
    out_path = Path("ixi_output")
    out_path.mkdir(exist_ok=True)
    
    nib.save(nib.Nifti1Image(f_stick_vol, affine), out_path / "ixi_bas_f_stick.nii.gz")
    nib.save(nib.Nifti1Image(out_vol, affine), out_path / "ixi_bas_params.nii.gz")
    
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
