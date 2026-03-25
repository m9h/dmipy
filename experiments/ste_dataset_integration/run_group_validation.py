import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import time
import os
import glob
import pandas as pd
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.gradients.aganj_estimator import fit_aganj_gradients

def run_group_validation():
    print("--- Aganj Gradient Estimator: Group Validation (10+ Subjects) ---")
    
    root_dir = "/home/mhough/Downloads/STE"
    # Find all subject directories STE*
    # specific structure: STE00_ExVivo, STE01, ...
    sub_dirs = sorted(glob.glob(os.path.join(root_dir, "STE*")))
    
    results_list = []
    
    # JIT Compile the fitter once (shape independent if we use slice)
    # We'll use a fixed slice shape (128x128). Usually datasets are registered/same dim.
    # We compile inside the loop or lazily.
    
    # Define Fitter
    # fit_aganj_gradients(params)
    
    for sub_path in sub_dirs:
        sub_name = os.path.basename(sub_path)
        print(f"\nProcessing {sub_name}...")
        
        # Locate Data
        # Structure varies? STE00 has 'STE' subdir.
        # Check if 'STE' subdir exists
        if os.path.exists(os.path.join(sub_path, "STE")):
            data_dir = os.path.join(sub_path, "STE")
        else:
            data_dir = sub_path
            
        nii_path = os.path.join(data_dir, "STE_degibbs_eddy.nii.gz")
        if not os.path.exists(nii_path):
            # Try default name
            nii_path = os.path.join(data_dir, "STE.nii.gz")
            
        if not os.path.exists(nii_path):
            print(f"  Skipping: Data not found in {data_dir}")
            continue
            
        # Bvecs/Bvals
        bvecs_path = os.path.join(data_dir, "bvecs.txt")
        bvals_path = os.path.join(data_dir, "bvals.txt")
        
        try:
            bvecs = np.loadtxt(bvecs_path).T
            if bvecs.shape[0] == 3: bvecs = bvecs.T
            bvals = np.loadtxt(bvals_path)
        except:
             print("  Warning: bvecs/bvals issues. Skipping.")
             continue
             
        # Load Nifti
        img = nib.load(nii_path)
        data = img.get_fdata() # (X, Y, Z, N)
        
        # Pick Middle Slice
        lz = data.shape[2]
        z_slice = lz // 2
        sl_data = data[:, :, z_slice, :]
        
        # Mask/Normalize S0
        b0_idx = bvals < 50
        s0 = np.mean(sl_data[..., b0_idx], axis=-1)
        s0 = np.maximum(s0, 1e-6)
        mask = s0 > 0.1 * np.max(s0)
        
        s_norm = sl_data / s0[..., None]
        s_norm_jax = jnp.array(s_norm)
        
        # Acquisition
        acq = JaxAcquisition(bvalues=jnp.array(bvals), gradient_directions=jnp.array(bvecs))
        
        # Fit (JIT)
        # Re-JIT for each subject to be safe with shapes (though usually 128x128)
        # Use tau=1.0 for all (approx)
        fit_fn = jax.vmap(jax.vmap(lambda s: fit_aganj_gradients(s, acq, tau=1.0), in_axes=0), in_axes=0)
        fit_compiled = jax.jit(fit_fn)
        
        t0 = time.time()
        res = fit_compiled(s_norm_jax)
        res.block_until_ready()
        t_fit = time.time() - t0
        
        # Analyze
        g_est = res[..., 6:9]
        
        # Compare with S0 Gradient
        grad_y, grad_x = np.gradient(s0)
        grad_ref = np.stack([grad_x, grad_y, np.zeros_like(grad_x)], axis=-1)
        
        # Compute Cosine Similarity
        norm_est = np.linalg.norm(g_est, axis=-1, keepdims=True) + 1e-9
        norm_ref = np.linalg.norm(grad_ref, axis=-1, keepdims=True) + 1e-9
        
        dir_est = g_est / norm_est
        dir_ref = grad_ref / norm_ref
        
        dot = np.sum(dir_est * dir_ref, axis=-1)
        # Filter mask
        mean_align = np.mean(np.abs(dot[mask]))
        mean_g_mag = np.mean(np.linalg.norm(g_est[mask], axis=-1))
        
        print(f"  Fit Time: {t_fit:.2f}s")
        print(f"  Alignment: {mean_align:.4f}")
        print(f"  Gradient Mag: {mean_g_mag:.4f}")
        
        results_list.append({
            "Subject": sub_name,
            "Alignment": mean_align,
            "GradientMag": mean_g_mag,
            "Time": t_fit
        })
    
    # Summary
    print("\n--- Group Summary ---")
    df = pd.DataFrame(results_list)
    print(df)
    
    mean_all = df["Alignment"].mean()
    print(f"\nOverall Mean Alignment: {mean_all:.4f}")
    
    df.to_csv("Aganj_Group_Validation_Results.csv", index=False)
    print("Saved Aganj_Group_Validation_Results.csv")

if __name__ == "__main__":
    run_group_validation()
