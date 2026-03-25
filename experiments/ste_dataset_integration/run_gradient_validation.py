import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import time
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.gradients.aganj_estimator import fit_aganj_gradients

def run_validation():
    print("--- Aganj Gradient Estimator Validation ---")
    
    # 1. Load Data
    data_path = "/home/mhough/Downloads/STE/STE00_ExVivo/STE/STE_degibbs_eddy.nii.gz"
    print(f"Loading {data_path}...")
    img = nib.load(data_path)
    data = img.get_fdata() # (X, Y, Z, N_dirs)
    affine = img.affine
    
    # 2. Load Bvecs/Bvals
    # Assuming standard STE b=1000/2000 structure or similar.
    # We'll use the bvals/bvecs from verifying script logic or similar default.
    # The STE dataset has bvals/bvecs files?
    # Let's check directories or assume standard structure.
    # Previous viewing showed `experiments/ste_dataset_integration/results` but not raw bvecs.
    # However `verify_ste_pipeline.py` loaded them. Let's assume adjacent files or hardcode for prototype.
    # Actually, `verify_ste_pipeline.py` created Dummy bvecs/bvals for benchmark? 
    # No, it says "Extract b=1000 shell" implies it read them.
    # Let's check `verify_ste_pipeline.py` content again if needed. 
    # Or just Assume b=1000 and simple directions? 
    # Wait, `verify_ste_pipeline.py` uses real data? 
    # "Created a pipeline verification script... loads STE data... Performs data loading"
    # It likely reads `STE_degibbs_eddy.bvecs` etc.
    # Let's assume standard names.
    
    import os
    base_dir = os.path.dirname(data_path)
    bvecs_path = os.path.join(base_dir, "bvecs.txt")
    bvals_path = os.path.join(base_dir, "bvals.txt")
    
    try:
        bvecs = np.loadtxt(bvecs_path).T # (N, 3) Check if transposing needed. bvecs.txt usually 3xN.
        if bvecs.shape[1] == 3 and bvecs.shape[0] != 3:
             # Already Nx3?
             pass
        elif bvecs.shape[0] == 3:
             bvecs = bvecs.T
             
        bvals = np.loadtxt(bvals_path)
    except:
        print("Warning: bvecs/bvals not found. Generating dummy (256 dirs).")
        # Generate 256 dummy dirs roughly matching data size
        n_dirs = data.shape[-1]
        bvecs = np.random.randn(n_dirs, 3)
        bvecs /= np.linalg.norm(bvecs, axis=1)[:,None]
        bvals = np.ones(n_dirs) * 1000.0
        bvals[0] = 0.0 # Standard b0
    
    # JaxAcquisition
    acq = JaxAcquisition(bvalues=jnp.array(bvals), gradient_directions=jnp.array(bvecs))
    
    # 3. Preprocess
    # Extract Slice Z=12 (Middle)
    # data: (128, 128, 25, N) -> (128, 128, N)
    slice_idx = 12
    sl_data = data[:, :, slice_idx, :]
    
    # Normalize S/S0 (using mean b0)
    b0_mask = bvals < 50
    s0 = np.mean(sl_data[..., b0_mask], axis=-1)
    s0 = np.maximum(s0, 1e-6)
    
    # Normalize
    # Handle broadcasting: (128, 128, N) / (128, 128, 1)
    s_norm = sl_data / s0[..., None]
    
    # Convert to JAX
    s_norm_jax = jnp.array(s_norm)
    
    # 4. Fit Aganj Gradients
    print("Fitting Aganj Gradients to slice (vmap)...")
    
    # Use tau=1.0s (STE diffusion time approx)
    # fit_aganj_gradients takes (N_dirs,) signal -> (9,) params.
    fit_fn = jax.vmap(jax.vmap(lambda s: fit_aganj_gradients(s, acq, tau=1.0), in_axes=0), in_axes=0)
    
    # JIT
    fit_fn_compiled = jax.jit(fit_fn)
    
    t0 = time.time()
    results = fit_fn_compiled(s_norm_jax)
    results.block_until_ready()
    duration = time.time() - t0
    print(f"Fit completed in {duration:.4f}s")
    
    # Extract Gradient Vector g = results[..., 6:9]
    g_est = results[..., 6:9] 
    g_mag = np.linalg.norm(g_est, axis=-1)
    print(f"Mean Estimated Gradient Magnitude: {np.mean(g_mag):.6f}")
    
    # 5. Reference Comparison
    grad_y, grad_x = np.gradient(s0)
    grad_ref = np.stack([grad_x, grad_y, np.zeros_like(grad_x)], axis=-1)
    grad_ref_mag = np.linalg.norm(grad_ref, axis=-1)
    print(f"Mean Reference Gradient Magnitude: {np.mean(grad_ref_mag):.6f}")
    
    # 5. Reference Comparison
    # Compute Gradient of S0 image using Sobel/Gradient
    # S0 shape: (128, 128)
    # grad_S0: (2, 128, 128) -> (dy, dx)
    
    grad_y, grad_x = np.gradient(s0)
    # Z-gradient is unknown in 2D slice, assume 0 for comparison?
    grad_ref = np.stack([grad_x, grad_y, np.zeros_like(grad_x)], axis=-1)
    
    # Normalize for direction comparison
    norm_est = np.linalg.norm(g_est, axis=-1, keepdims=True) + 1e-9
    norm_ref = np.linalg.norm(grad_ref, axis=-1, keepdims=True) + 1e-9
    
    dir_est = g_est / norm_est
    dir_ref = grad_ref / norm_ref
    
    # Dot product (Cosine Similarity)
    # Mask out background (S0 approx 0)
    mask = s0 > 0.1 * np.max(s0)
    
    dot_prod = np.sum(dir_est * dir_ref, axis=-1)
    mean_align = np.mean(np.abs(dot_prod[mask])) # Abs because gradient sign might flip depending on definition
    
    print(f"Mean Directional Alignment with S0 Gradient: {mean_align:.4f}")
    
    # 6. Save Results
    out_img = nib.Nifti1Image(np.array(g_est), affine)
    nib.save(out_img, "STE_Aganj_Gradients_z12.nii.gz")
    print("Saved STE_Aganj_Gradients_z12.nii.gz")

if __name__ == "__main__":
    run_validation()
