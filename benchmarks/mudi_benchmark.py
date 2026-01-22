import argparse
import numpy as np
import nibabel as nib
import jax.numpy as jnp
from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.io.gradients import read_bvals_bvecs
import sys

def load_mudi_data(nifti_path, bval_path, bvec_path):
    """
    Loads MUDI data from NIfTI and bval/bvec files.
    """
    print(f"Loading data from {nifti_path}...")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs=bvecs)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of volumes: {len(bvals)}")
    
    return data, gtab, bvals, bvecs

def subsample_data(data, bvals, bvecs):
    """
    Subsamples the full dataset to a clinical subset (b=1000, b=2000).
    Returns training data/gtab and testing data/gtab (complement).
    """
    # Define clinical subset criteria
    # Typically includes b=0 and specific shells
    # For MUDI, let's assume standard multi-shell: b ~ 1000 and b ~ 2000
    # And keep b0s
    
    mask_b0 = bvals < 50
    mask_b1000 = (bvals > 900) & (bvals < 1100)
    mask_b2000 = (bvals > 1900) & (bvals < 2100)
    
    # We want a subset of these. 
    # MUDI has many directions. Let's take a fixed number if available, or all valid ones if fewer.
    # The prompt says: "b=1000 with 30 dirs, b=2000 with 60 dirs"
    
    indices_b0 = np.where(mask_b0)[0]
    indices_b1000 = np.where(mask_b1000)[0]
    indices_b2000 = np.where(mask_b2000)[0]
    
    # specific selection
    rng = np.random.RandomState(42) # Fixed seed for reproducibility
    
    # Keep all b0s (or a subset if there are too many, but usually good to keep)
    train_indices_b0 = indices_b0
    
    # Select 30 from b1000
    if len(indices_b1000) >= 30:
        train_indices_b1000 = rng.choice(indices_b1000, 30, replace=False)
    else:
        train_indices_b1000 = indices_b1000
        
    # Select 60 from b2000
    if len(indices_b2000) >= 60:
        train_indices_b2000 = rng.choice(indices_b2000, 60, replace=False)
    else:
        train_indices_b2000 = indices_b2000
        
    train_indices = np.concatenate([train_indices_b0, train_indices_b1000, train_indices_b2000])
    train_indices.sort()
    
    # Test indices are everything else
    all_indices = np.arange(len(bvals))
    test_indices = np.setdiff1d(all_indices, train_indices)
    
    print(f"Training volumes: {len(train_indices)} (b0: {len(train_indices_b0)}, b1000: {len(train_indices_b1000)}, b2000: {len(train_indices_b2000)})")
    print(f"Testing volumes: {len(test_indices)}")
    
    # Create training set
    train_data = data[..., train_indices]
    train_bvals = bvals[train_indices]
    train_bvecs = bvecs[train_indices]
    train_gtab = gradient_table(train_bvals, bvecs=train_bvecs)
    
    # Create testing set
    test_data = data[..., test_indices]
    test_bvals = bvals[test_indices]
    test_bvecs = bvecs[test_indices]
    test_gtab = gradient_table(test_bvals, bvecs=test_bvecs)
    
    return train_data, train_gtab, test_data, test_gtab

from dipy.reconst.shm import sph_harm_ind_list, real_sh_descoteaux
import numpy as np

def run_baseline_shm(train_data, train_gtab, test_gtab):
    """
    Fits a Spherical Harmonics model to the training data and predicts the test data.
    """
    # Calculate SH order
    # For ~30-90 points, order 6 or 8 is appropriate.
    sh_order = 8
    
    # 1. Compute Design Matrix for Training Data
    # dipy uses (m, n) indexing or (l, m).
    # real_sym_sh_basis(sh_order, theta, phi)
    # We need spherical coordinates from gtab
    
    train_bvecs = train_gtab.bvecs
    # Handle b0s? SH usually fits on a shell.
    # MUDI has multiple shells. Simple SH interpolation often fits mask-free or per-shell.
    # But usually "Spherical Harmonics Interpolation" implies fitting one SH to the data projected on sphere.
    # However, signal varies with b-value. 
    # If we treat it as a single shell (normalizing by b0), we ignore b-value dependence.
    # For a "Baseline", typically one shell is used, or a multi-shell SH (MSSH).
    # The prompt says "b=1000 with 30 dirs, b=2000 with 60 dirs".
    # A simple SH baseline might fit separate SHs for each shell, or just one if we ignore b-value (bad idea).
    # I will implement per-shell SH interpolation.
    
    predicted_signal = np.zeros((train_data.shape[0], train_data.shape[1], train_data.shape[2], len(test_gtab.bvals)))
    
    # Identify shells in test data
    # We will predict each test volume based on the SH fit of the CLOSEST shell in training data.
    
    unique_b_train = np.unique(np.round(train_gtab.bvals, -2)) # round to nearest 100
    unique_b_test = np.round(test_gtab.bvals, -2)
    
    print(f"Training shells: {unique_b_train}")
    
    for ub in np.unique(unique_b_test):
        # find matching training shell
        # If b=0, just mean of training b0s?
        if ub < 50:
             # Average all training b0s
             mask_train = train_gtab.bvals < 50
             if np.any(mask_train):
                 mean_b0 = np.mean(train_data[..., mask_train], axis=-1)
                 # Assign to all test b0 indices
                 mask_test = unique_b_test < 50
                 # predictions need to be placed in correct indices
                 # test_gtab indices match predicted_signal last dim? Yes.
                 idx_test = np.where(mask_test)[0]
                 for idx in idx_test:
                     predicted_signal[..., idx] = mean_b0
             continue

        # For DW shells
        # Find closest training shell
        dist = np.abs(unique_b_train - ub)
        closest_train_b = unique_b_train[np.argmin(dist)]
        
        # Fit SH on this training shell
        mask_train = np.abs(train_gtab.bvals - closest_train_b) < 100
        if np.sum(mask_train) == 0:
            print(f"Warning: No training data for shell {ub}, skipping...")
            continue
            
        train_shell_data = train_data[..., mask_train]
        train_shell_bvecs = train_gtab.bvecs[mask_train]
        
        # Generate basis
        _, theta_train, phi_train = cart2sphere(train_shell_bvecs[:, 0], train_shell_bvecs[:, 1], train_shell_bvecs[:, 2])
        B_train, m, n = real_sh_descoteaux(sh_order, theta_train, phi_train)
        
        # Regularized fit: C = (B.T B + lambda I)^-1 B.T S
        # lambda = 0.006 is standard default
        lambda_reg = 0.006
        inv_B = np.linalg.pinv(B_train.T @ B_train + lambda_reg * np.eye(B_train.shape[1])) @ B_train.T
        
        # Fit coeffs: (X, Y, Z, n_coeffs)
        coeffs = np.dot(train_shell_data, inv_B.T)
        
        # Predict on test
        mask_test = np.abs(unique_b_test - ub) < 50
        idx_test = np.where(mask_test)[0]
        
        test_shell_bvecs = test_gtab.bvecs[idx_test]
        _, theta_test, phi_test = cart2sphere(test_shell_bvecs[:, 0], test_shell_bvecs[:, 1], test_shell_bvecs[:, 2])
        B_test, _, _ = real_sh_descoteaux(sh_order, theta_test, phi_test)
        
        # Predict: S = C @ B_test.T
        # coeffs: (..., n_c), B_test: (n_test, n_c)
        # S: (..., n_test)
        pred_shell = np.dot(coeffs, B_test.T)
        
        # Assign
        for i, original_idx in enumerate(idx_test):
            predicted_signal[..., original_idx] = pred_shell[..., i]

    return predicted_signal

def evaluate_metrics(predicted, ground_truth):
    """
    Calculates MSE between predicted and ground truth.
    Also converts to JAX arrays.
    """
    # Convert to JAX arrays
    predicted_jax = jnp.array(predicted)
    ground_truth_jax = jnp.array(ground_truth)
    
    # Calculate MSE
    # Averaging over all dimensions
    mse = jnp.mean((predicted_jax - ground_truth_jax) ** 2)
    
    return mse, predicted_jax, ground_truth_jax

def main():
    parser = argparse.ArgumentParser(description="MUDI Benchmark Script")
    parser.add_argument("nifti_file", help="Path to MUDI NIfTI file (4D)")
    parser.add_argument("bval_file", help="Path to bval file")
    parser.add_argument("bvec_file", help="Path to bvec file")
    parser.add_argument("--mask_file", help="Optional mask file", default=None)
    
    args = parser.parse_args()
    
    # 1. Data Loading
    data, full_gtab, bvals, bvecs = load_mudi_data(args.nifti_file, args.bval_file, args.bvec_file)
    
    # Apply mask if provided
    if args.mask_file:
        print(f"Applying mask from {args.mask_file}...")
        mask_img = nib.load(args.mask_file)
        mask = mask_img.get_fdata() > 0
        # Expand mask to 4D for multiplication if needed, or just mask indices
        # For simplicity, we'll zero out background or just fit inside mask
        # Dipy fits usually handle masking by only fitting voxels.
        # But 'fit.predict' outputs full volume. 
        # Let's apply mask to data to zero out background noise
        data = data * mask[..., np.newaxis]
    
    # 2. Simulation of Sparse Input (Subsampler)
    train_data, train_gtab, test_data_gt, test_gtab = subsample_data(data, bvals, bvecs)
    
    # 3. Baseline Model (SH Interpolation)
    predicted_signal = run_baseline_shm(train_data, train_gtab, test_gtab)
    
    # 4. Evaluation (MSE) & JAX Compatibility
    mse, pred_jax, gt_jax = evaluate_metrics(predicted_signal, test_data_gt)
    
    print("="*40)
    print(f"Benchmark Results:")
    print(f"Mean Squared Error (MSE): {mse}")
    print("="*40)
    
    print(f"JAX Array Shapes: Pred {pred_jax.shape}, GT {gt_jax.shape}")

if __name__ == "__main__":
    main()
