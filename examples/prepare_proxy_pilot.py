
import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
from dmipy_jax.models.ball_stick import BallStick
from dmipy_jax.fitting.optimization import VoxelFitter
from dmipy_jax.acquisition import JaxAcquisition

def get_proxy_subjects(bids_root, n_subjects=10):
    """
    Finds first N subjects (proxy for 18-22y filtering).
    """
    subs = sorted(glob.glob(os.path.join(bids_root, 'sub-*')))
    return [os.path.basename(s) for s in subs[:n_subjects]]

def load_dwi_data(subject_dir):
    """
    Loads dwi.nii.gz, bval, bvec for a subject (ses-02 hardcoded for ds004910 proxy).
    """
    # Try different sessions or recursive search
    dwi_glob = glob.glob(os.path.join(subject_dir, 'ses-*', 'dwi', '*_dwi.nii.gz'))
        
    if not dwi_glob:
        print(f"No DWI data found for {subject_dir}")
        return None
        
    dwi_file = dwi_glob[0] # Take first session found
    print(f"Loading {dwi_file}")
    
    bval_file = dwi_file.replace('.nii.gz', '.bval')
    bvec_file = dwi_file.replace('.nii.gz', '.bvec')
    
    if not (os.path.exists(bval_file) and os.path.exists(bvec_file)):
        print(f"Missing bvals/bvecs for {dwi_file}")
        return None
        
    img = nib.load(dwi_file)
    data = img.get_fdata()
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T # (N, 3)
    
    return data, bvals, bvecs, img.affine

def fit_ground_truth(data, bvals, bvecs, mask=None):
    """
    Fits BallStick to multi-shell data to estimate f_iso.
    """
    # 1. Prepare Data
    if mask is None:
        # Simple intensity mask
        mask = np.mean(data, axis=-1) > np.percentile(data, 10) 
        
    # Flatten mask
    # OPTIMIZATION: Subsample for Pilot speed (take every 10th voxel in each dim ~ 1/1000th of volume)
    # Actually, simple subsampling might miss brain if not careful.
    # Let's just create a strided view.
    mask_strided = np.zeros_like(mask)
    # Use 5% of valid voxels
    valid_coords = np.argwhere(mask)
    if len(valid_coords) > 20000:
        # Take random subset of 20,000 voxels for training pilot
        np.random.shuffle(valid_coords)
        subset = valid_coords[:20000]
        mask_strided[subset[:, 0], subset[:, 1], subset[:, 2]] = 1
        mask = mask_strided
        print("Subsampled to 20,000 voxels for speedy pilot.")

    data_flat = data.reshape(-1, data.shape[-1])
    mask_flat = mask.reshape(-1)
    
    # Filter voxels to fit
    valid_indices = np.where(mask_flat)[0]
    data_valid = jnp.array(data_flat[valid_indices])
    
    print(f"Fitting {len(valid_indices)} voxels...")
    
    # 2. Setup Model (BallStick)
    bs = BallStick()
    
    # 3. Setup Fitter
    bounds = [
        (0.0, jnp.pi),       # theta
        (-jnp.pi, jnp.pi),   # phi
        (0.0, 1.0)           # f_stick
    ]
    fitter = VoxelFitter(bs, bounds)
    
    # 4. Acquisition
    acq = JaxAcquisition(bvalues=jnp.array(bvals), gradient_directions=jnp.array(bvecs))
    
    # 5. Initialization
    n_valid = len(valid_indices)
    init_params = jnp.column_stack([
         jnp.full(n_valid, 1.57), 
         jnp.full(n_valid, 0.0),  
         jnp.full(n_valid, 0.5)   
    ])
    
    # 6. Fit
    # Batch processing to avoid OOM on full brain
    batch_size = 50000
    n_batches = (n_valid + batch_size - 1) // batch_size
    
    fit_fn = jax.jit(jax.vmap(fitter.fit, in_axes=(0, None, 0)))
    
    all_params = []
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_valid)
        if start >= end: break
        
        batch_data = data_valid[start:end]
        batch_init = init_params[start:end]
        
        p, _ = fit_fn(batch_data, acq, batch_init)
        all_params.append(p)
        print(f"Batch {i+1}/{n_batches} done.")
        
    params_opt = jnp.concatenate(all_params, axis=0)
    
    # 7. Extract Free Water (f_iso = 1 - f_stick)
    f_stick = params_opt[:, 2]
    f_iso = 1.0 - f_stick
    
    # 8. Reconstruct Map
    f_iso_map = np.zeros(mask_flat.shape)
    f_iso_map[valid_indices] = np.array(f_iso)
    f_iso_map = f_iso_map.reshape(data.shape[:-1])
    
    return f_iso_map, mask

def prepare_proxy_pilot(bids_root, output_dir="data/proxy_pilot"):
    os.makedirs(output_dir, exist_ok=True)
    
    subjects = get_proxy_subjects(bids_root)
    print(f"Processing subjects: {subjects}")
    
    for sub in subjects:
        sub_dir = os.path.join(bids_root, sub)
        print(f"Processing {sub}...")
        res = load_dwi_data(sub_dir)
        if res is None: continue
        
        data, bvals, bvecs, affine = res
        
        # 1. Fit Ground Truth (Multi-Shell)
        f_iso_gt, mask = fit_ground_truth(data, bvals, bvecs)
        
        # 2. Create Single-Shell Input (b=1000)
        # B-values in ds004910 are likely ~1000, ~2000
        # Check range
        print(f"B-values found: {np.unique(np.round(bvals, -2))}")
        
        b1000_mask = (bvals > 800) & (bvals < 1200)
        b0_mask = (bvals < 50)
        
        keep_indices = np.where(b1000_mask | b0_mask)[0]
        
        if len(keep_indices) < 10:
             print("Warning: Not enough b=1000 shells found. Skipping.")
             continue

        data_input = data[..., keep_indices]
        bvals_input = bvals[keep_indices]
        bvecs_input = bvecs[keep_indices]
        
        # 3. Save
        out_file = os.path.join(output_dir, f"{sub}_training.npz")
        np.savez(out_file, 
                 signals=data_input, 
                 bvals=bvals_input, 
                 bvecs=bvecs_input, 
                 f_iso_gt=f_iso_gt,
                 mask=mask)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str, required=True, help="Path to Proxy BIDS dataset")
    parser.add_argument("--output_dir", type=str, default="data/proxy_pilot")
    args = parser.parse_args()
    
    prepare_proxy_pilot(args.bids_root, args.output_dir)
