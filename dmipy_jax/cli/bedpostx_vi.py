
import os
import argparse
import sys
import jax
import jax.numpy as jnp
import nibabel as nib
import numpy as np
from dmipy_jax.bayesian.inference import fit_batch_vi

def main():
    parser = argparse.ArgumentParser(description="dmipy-jax Bayesian Inference (Bedpostx-VI)")
    parser.add_argument("data_dir", help="Directory containing data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz")
    parser.add_argument("-n", "--fibres", type=int, default=3, help="Maximum number of fibers per voxel")
    parser.add_argument("-w", "--ardweight", type=float, default=1.0, help="ARD weight (higher = sparser)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of VI optimization steps")
    parser.add_argument("--out", help="Output directory", default=None)
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    # Setup paths
    dwi_path = os.path.join(data_dir, "data.nii.gz")
    mask_path = os.path.join(data_dir, "nodif_brain_mask.nii.gz")
    bvals_path = os.path.join(data_dir, "bvals")
    bvecs_path = os.path.join(data_dir, "bvecs")
    
    # Check inputs
    if not all(os.path.exists(p) for p in [dwi_path, mask_path, bvals_path, bvecs_path]):
        print(f"Error: Missing files in {data_dir}. Expected data.nii.gz, mask, bvals, bvecs.")
        sys.exit(1)
        
    out_dir = args.out if args.out else os.path.join(data_dir, "dmipy_vi.bedpostX")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    img = nib.load(dwi_path)
    data = img.get_fdata()
    mask = nib.load(mask_path).get_fdata() > 0
    affine = img.affine
    
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path).T # dmipy usually expects (N, 3)
    
    # Flatten mask for processing
    # Only process mask voxels
    mask_indices = np.where(mask)
    valid_data = data[mask_indices] # (N_valid, N_diff)
    
    print(f"Processing {valid_data.shape[0]} voxels with JAX...")
    
    # Move to JAX
    bvals_j = jnp.array(bvals)
    bvecs_j = jnp.array(bvecs)
    data_j = jnp.array(valid_data)
    
    # Run Inference
    # Batch size handling: JAX works best with consistent batch sizes.
    # We might need to chunk if the number of voxels is huge (millions).
    # For now, let's process in chunks of 5000 to be safe on VRAM.
    chunk_size = 5000
    n_voxels = data_j.shape[0]
    results_list = []
    
    rng_key = jax.random.PRNGKey(42)
    
    for i in range(0, n_voxels, chunk_size):
        end = min(i + chunk_size, n_voxels)
        print(f"  Chunk {i}-{end} / {n_voxels}")
        batch_data = data_j[i:end]
        rng_key, subkey = jax.random.split(rng_key)
        
        # Run VI
        batch_res, _ = fit_batch_vi(
            subkey, 
            batch_data, 
            bvals_j, 
            bvecs_j, 
            n_fibers=args.fibres, 
            ard_weight=args.ardweight,
            num_steps=args.steps
        )
        results_list.append(batch_res)
        
    # Reconstruct Volume
    # We need to map the 1D results back to the 3D volume
    # Result keys will look like 'd', 'S0', 'f', 'v_raw' (based on AutoNormal)
    # Actually AutoNormal returns latent values. We asked for 'guide.median'
    # which estimates the constrained values if we used constrained sites?
    # No, AutoNormal works on unconstrained space usually.
    # We need to handle the transforms or look at the median values.
    # For simple parity, let's just inspect the keys.
    # Wait, 'guide.median(params)' returns the median of the distribution in site space.
    
    # Concatenate results
    # We assume the structure is consistent
    combined_res = {}
    first_res = results_list[0]
    for k in first_res.keys():
        combined_res[k] = np.concatenate([r[k] for r in results_list], axis=0) # back to numpy
        
    print("Saving outputs...")
    
    # Helper to save 3D volume
    def save_vol(name, values_1d):
        vol = np.zeros(mask.shape + values_1d.shape[1:], dtype=np.float32)
        vol[mask_indices] = values_1d
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(out_dir, name))

    # Save S0
    save_vol("mean_S0.nii.gz", combined_res['S0'][:, None])
    
    # Save d
    save_vol("mean_d.nii.gz", combined_res['d'][:, None])
    
    # Save Fractions and Dyads
    f = combined_res['f'] # (N, K)
    v_raw = combined_res['v_raw'] # (N, K, 3)
    
    # Normalize v
    norm = np.linalg.norm(v_raw, axis=-1, keepdims=True)
    v = v_raw / (norm + 1e-9)
    
    for i in range(args.fibres):
        # Bedpostx uses 1-based indexing for filenames
        idx = i + 1
        save_vol(f"mean_f{idx}samples.nii.gz", f[:, i:i+1])
        save_vol(f"dyads{idx}.nii.gz", v[:, i, :])
        
    print(f"Done. Outputs in {out_dir}")

if __name__ == "__main__":
    main()
