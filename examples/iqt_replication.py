"""
IQT Replication Script (Alexander et al., 2017)
Replicates the Image Quality Transfer (IQT) experimental setup using HCP data.

Usage:
    python examples/iqt_replication.py --subject <subject_id> --data_root <path>

"""

import os
import argparse
import numpy as np
import jax.numpy as jnp
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.image import extract_patches_2d
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs

# dmipy-jax imports
from dmipy_jax.io.hcp import load_hcp_subject

def compute_psnr(gt, recon, dynamic_range=None):
    """
    Computes Peak Signal-to-Noise Ratio.
    """
    mse = np.mean((gt - recon) ** 2)
    if mse == 0:
        return float('inf')
    
    if dynamic_range is None:
        dynamic_range = gt.max() - gt.min()
        
    return 20 * np.log10(dynamic_range / np.sqrt(mse))

def compute_ssim(gt, recon, dynamic_range=None):
    """
    Computes Structural Similarity Index (SSIM).
    Wrapper around skimage.metrics.structural_similarity if available,
    else simplified implementation.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        if dynamic_range is None:
            dynamic_range = gt.max() - gt.min()
        return ssim(gt, recon, data_range=dynamic_range)
    except ImportError:
        print("skimage not found, skipping SSIM")
        return 0.0

def compute_dti_metrics(data, bvals, bvecs, mask=None):
    """
    Computes FA and MD using Dipy.
    Data: (H, W, D, N)
    """
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    
    # Fit
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)
        
    tenfit = tenmodel.fit(data, mask=mask)
    
    FA = tenfit.fa
    MD = tenfit.md
    
    return FA, MD

def extract_features_and_targets(low_res_data, high_res_data, patch_size=5, subsample=1.0):
    """
    Extracts patches from Low-Res (Upsampled) data and corresponding center voxels from High-Res.
    
    Args:
        low_res_data: (H, W, D, C) - The input (simulated low-res, upsampled to match grid)
        high_res_data: (H, W, D, C) - The target (ground truth)
        patch_size: int (cubic patch size, e.g. 5x5x5)
    
    Returns:
        X: (N_samples, n_features)
        y: (N_samples, n_channels)
    """
    rad = patch_size // 2
    
    # Valid region
    H, W, D, C = low_res_data.shape
    
    # For simplicity in this benchmark, we'll take a random subset of voxels to form the train set
    # because extracting ALL patches is memory intensive.
    
    # Create a mask of valid center pixels
    valid_mask = np.zeros((H, W, D), dtype=bool)
    valid_mask[rad:H-rad, rad:W-rad, rad:D-rad] = True
    
    # Indices
    coords = np.argwhere(valid_mask)
    
    if subsample < 1.0:
        n_samples = int(len(coords) * subsample)
        indices = np.random.choice(len(coords), n_samples, replace=False)
        coords = coords[indices]
    
    X = []
    y = []
    
    print(f"Extracting patches from {len(coords)} locations...")
    
    # Optimize this loop or use specialized patch extraction if needed.
    # For now, simple loop is fine for benchmarking correctness.
    
    # Flatten spatial dims for easier indexing? No, keep 3D.
    # A faster way: pre-extract patches using a sliding window view, but that consumes RAM.
    # Let's just loop for the prototype.
    
    for idx, (x, y_coord, z) in enumerate(coords):
        # Patch from Input
        patch = low_res_data[x-rad:x+rad+1, y_coord-rad:y_coord+rad+1, z-rad:z+rad+1, :]
        X.append(patch.flatten())
        
        # Target: Center voxel from GT
        target = high_res_data[x, y_coord, z, :]
        y.append(target)
        
    return np.array(X), np.array(y)

def reconstruct_volume(low_res_data, model, patch_size=5):
    """
    Reconstructs the high-res volume using the trained RF model.
    """
    H, W, D, C = low_res_data.shape
    rad = patch_size // 2
    
    recon = np.zeros_like(low_res_data)
    # Copy borders from input as fallback
    recon[:] = low_res_data[:]
    
    valid_mask = np.zeros((H, W, D), dtype=bool)
    valid_mask[rad:H-rad, rad:W-rad, rad:D-rad] = True
    coords = np.argwhere(valid_mask)
    
    print(f"Reconstructing {len(coords)} voxels...")
    
    # Batch prediction to save memory/time
    batch_size = 10000
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        X_batch = []
        for x, y_coord, z in batch_coords:
            patch = low_res_data[x-rad:x+rad+1, y_coord-rad:y_coord+rad+1, z-rad:z+rad+1, :]
            X_batch.append(patch.flatten())
        
        y_pred = model.predict(X_batch)
        
        for j, (x, y_coord, z) in enumerate(batch_coords):
            recon[x, y_coord, z, :] = y_pred[j]
            
    return recon

def main():
    parser = argparse.ArgumentParser(description="IQT Replication Benchmark")
    parser.add_argument("--subject", type=str, default="100307", help="HCP Subject ID")
    parser.add_argument("--data_root", type=str, default="data/HCP", help="Data root directory")
    parser.add_argument("--downsample_factor", type=float, default=2.0, help="Downsampling factor (1.25 -> 2.5 is factor 2)")
    parser.add_argument("--rf_trees", type=int, default=10, help="Number of trees in RF")
    parser.add_argument("--rf_depth", type=int, default=10, help="Max depth of RF")
    parser.add_argument("--training_subsample", type=float, default=0.01, help="Fraction of voxels to use for training (for speed)")
    parser.add_argument("--debug", action="store_true", help="Run with small subset of slices")
    
    args = parser.parse_args()
    
    print(f"Starting IQT Replication for Subject {args.subject}")
    
    if args.debug:
        print("DEBUG MODE: Using small mock dataset (Synthetic Crossing Fibers)")
        # Generate synthetic data
        from dipy.sims.voxel import multi_tensor
        from dipy.data import get_sphere
        from dipy.core.gradients import gradient_table

        # 1. Define Acquisition SCheme
        # 64 directions, b=1000
        sphere = get_sphere('symmetric724')
        bvecs = sphere.vertices[:64]
        bvals = np.ones(64) * 1000
        bvals = np.insert(bvals, 0, 0)
        bvecs = np.insert(bvecs, 0, [0, 0, 0], axis=0)
        gtab = gradient_table(bvals, bvecs)
        
        # 2. Define Tissue (Crossing Fibers)
        mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
        angles = [(0, 0), (90, 0)] # 90 degrees crossing
        fractions = [50, 50]
        
        # 3. Generate Volume (20x20x20)
        # We make a simple pattern: Cross in center, single fibers outer
        S0 = 100
        vol_shape = (20, 20, 20, len(bvals))
        data_dwi = np.zeros(vol_shape)
        
        print("Generating synthetic phantom...")
        # Fill with crossing
        signal_cross, _ = multi_tensor(gtab, mevals, S0=S0, angles=angles, fractions=fractions, snr=None)
        
        # Fill with single fiber 1
        signal_f1, _ = multi_tensor(gtab, mevals[:1], S0=S0, angles=angles[:1], fractions=[100], snr=None)
        
        # Fill center with cross
        data_dwi[:, :, :, :] = signal_f1
        data_dwi[5:15, 5:15, 5:15, :] = signal_cross
        
        # Add Rician noise
        from dipy.sims.voxel import add_noise
        data_dwi = add_noise(data_dwi, snr=30, S0=S0, noise_type='rician')
        
        # Create SimpleAcquisitionScheme-like object for consistency
        class MockScheme:
            bvalues = bvals
            gradient_directions = bvecs
        scheme = MockScheme()
        data_t1 = np.random.rand(20, 20, 20) # Dummy T1
        
        print(f"Mock Data Shape: {data_dwi.shape}")
        
    else:
        # 1. Load Data
        try:
            # load_hcp_subject returns jax arrays, convert to numpy for sklearn
            data_dwi, scheme, data_t1 = load_hcp_subject(args.subject, args.data_root)
            data_dwi = np.array(data_dwi) # (H, W, D, N)
            print(f"Original Data Shape: {data_dwi.shape}")
        except Exception as e:
            print(f"Failed to load HCP data: {e}")
            print("Try running with --debug to use synthetic data verification.")
            return

    # Split into Train (Top Half) and Test (Bottom Half) along Z axis


    # Split into Train (Top Half) and Test (Bottom Half) along Z axis
    z_mid = data_dwi.shape[2] // 2
    data_train = data_dwi[:, :, :z_mid, :]
    data_test = data_dwi[:, :, z_mid:, :]
    
    print(f"Train Partition: {data_train.shape}")
    print(f"Test Partition: {data_test.shape}")
    
    # 2. Degradation (Downsample -> Upsample)
    # We downsample by factor, then upsample back to original grid to simulate "Low Res on High Res Grid"
    # This acts as the feature input.
    
    print("Simulating Low-Res acquisition...")
    # Downsample
    # Order 3 = cubic spline
    zoom_factor = 1.0 / args.downsample_factor
    # Don't zoom channel dim (last dim)
    # We iterate over volumes or use ndimage on 3D parts
    
    def degradate_and_upsample(volume_4d):
        low_res = np.zeros_like(volume_4d)
        
        for i in range(volume_4d.shape[3]):
            vol = volume_4d[..., i]
            # Downsample
            vol_lr = zoom(vol, zoom_factor, order=3)
            # Upsample back
            # Calculate zoom back to match original shape exactly
            zoom_back = np.array(vol.shape) / np.array(vol_lr.shape)
            vol_interp = zoom(vol_lr, zoom_back, order=3)
            
            # Allow for minor shape mismatch from rounding
            # Crop or pad? zoom typically handles it but let's be safe
            # Actually simplest to just resize to exact target shape if supported?
            # ndimage.zoom doesn't take target shape.
            
            # Let's map into the buffer
            d, h, w = vol_interp.shape
            D, H, W = low_res.shape[:3]
            
            # Simple center crop/pad if needed, but usually it matches closely
            # For this prototype we assume it matches or we fix it.
            # Using skimage.transform.resize is safer for fixed output shape
            # but let's stick to scipy zoom for now.
            
            # Assign (with simple bounds check to avoid crash)
            d_end = min(d, D)
            h_end = min(h, H)
            w_end = min(w, W)
            low_res[:d_end, :h_end, :w_end, i] = vol_interp[:d_end, :h_end, :w_end]
            
        return low_res

    print("Processing Train Set...")
    input_train = degradate_and_upsample(data_train)
    print("Processing Test Set...")
    input_test = degradate_and_upsample(data_test)
    
    # 3. Train Random Forest (Baseline Oracle)
    print("Extracting features for Random Forest...")
    X_train, y_train = extract_features_and_targets(
        input_train, 
        data_train, 
        patch_size=3, # Use 3x3x3 for speed in this demo, paper uses 2x2x2 up to 5x5x5
        subsample=args.training_subsample
    )
    print(f"Training Samples: {X_train.shape[0]}")
    
    print("Fitting Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=args.rf_trees, 
        max_depth=args.rf_depth,
        n_jobs=-1,
        verbose=1
    )
    rf.fit(X_train, y_train)
    
    # 4. Predict on Test Set
    print("Reconstructing Test Set with IQT...")
    iqt_recon = reconstruct_volume(input_test, rf, patch_size=3)
    
    # 5. Metrics
    print("Computing Metrics...")
    
    # Needed for DTI
    # Use bvals/bvecs from acquisition scheme if available in simple form
    # We loaded data_dwi, scheme, data_t1.
    # We need raw bvals/bvecs. 
    # The `load_hcp_subject` returns `scheme` which is a `SimpleAcquisitionScheme`.
    # It has `bvalues` and `gradient_directions`.
    bvals = np.array(scheme.bvalues)
    bvecs = np.array(scheme.gradient_directions)
    
    print("Computing DTI maps for GT...")
    fa_gt, md_gt = compute_dti_metrics(data_test, bvals, bvecs)
    
    print("Computing DTI maps for Baseline (Spline)...")
    fa_bl, md_bl = compute_dti_metrics(input_test, bvals, bvecs)
    
    print("Computing DTI maps for IQT (RF)...")
    fa_iqt, md_iqt = compute_dti_metrics(iqt_recon, bvals, bvecs)
    
    # PSNR/SSIM
    # On FA
    psnr_fa_bl = compute_psnr(fa_gt, fa_bl)
    psnr_fa_iqt = compute_psnr(fa_gt, fa_iqt)
    ssim_fa_bl = compute_ssim(fa_gt, fa_bl)
    ssim_fa_iqt = compute_ssim(fa_gt, fa_iqt)
    
    print("\n=== Results ===")
    print(f"Baseline (Spline): PSNR(FA)={psnr_fa_bl:.2f}, SSIM(FA)={ssim_fa_bl:.4f}")
    print(f"IQT (RF)         : PSNR(FA)={psnr_fa_iqt:.2f}, SSIM(FA)={ssim_fa_iqt:.4f}")
    
    if psnr_fa_iqt > psnr_fa_bl:
        print("SUCCESS: IQT improved over baseline.")
    else:
        print("WARNING: IQT did not improve over baseline (check hyperparameters).")

    # Save outputs if needed
    # (implied)
    
if __name__ == "__main__":
    main()
