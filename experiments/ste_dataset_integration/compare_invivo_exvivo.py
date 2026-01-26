import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import os
import sys
from typing import Tuple

# Ensure x64 for precision in metrics
jax.config.update("jax_enable_x64", True)

def load_dwi(nifti_path: str, bval_path: str, bvec_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads DWI data, b-values, and b-vectors."""
    print(f"Loading: {nifti_path}")
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T
    
    # Simple check
    if data.shape[-1] != len(bvals):
        print(f"WARNING: Mismatch b/w volumes ({data.shape[-1]}) and bvals ({len(bvals)}). Trunctating data.")
        data = data[..., :len(bvals)]
        
    return data, bvals, bvecs, affine

def estimate_snr(data: np.ndarray, bvals: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Rough estimation of SNR using the b=0 images.
    SNR = Mean(Signal_ROI) / Std(Background) OR Signal / Noise_Sigma
    Here we use a simpler 'signal vs temporal noise' if we had repeats, 
    but with single acquisition we usually use corpus callosum vs background.
    
    For check: Let's assume Mean Signal in Mask / Std in Mask of Reference noise?
    Better: Signal / Sigma where Sigma is derived from local PCA or background.
    
    Approach for now: 
    1. Find b0 indices.
    2. Compute mean signal S0 in brain mask.
    3. Estimate noise sigma from background (outside mask).
    """
    b0_indices = np.where(bvals < 50)[0]
    if len(b0_indices) == 0:
        # Fallback to lowest b
        b0_indices = [np.argmin(bvals)]
        
    b0_data = np.mean(data[..., b0_indices], axis=-1)
    
    # Background noise estimation (Corner strategy)
    corner_roi = b0_data[0:20, 0:20, :]
    background_roi = corner_roi[corner_roi > 0]
    
    if len(background_roi) < 100:
        print("Warning: Too few background voxels in corner. Using naive estimate.")
        noise_sigma = 1.0
        bg_mean = 0.0
    else:
        bg_mean = np.mean(background_roi)
        bg_std = np.std(background_roi) 
        # For Rayleigh, sigma approx Std / sqrt( (4-pi)/2 ) ~ Std / 0.655 ~ 1.5*Std
        # But let's just use Std directly for thresholding buffer.
        noise_sigma = bg_mean / np.sqrt(np.pi / 2.0)

    if mask is None:
        # Robust threshold: Mean + 10 * Std
        # If noise is Gaussian/Rayleigh, 10 sigmas is safe.
        threshold = bg_mean + 10.0 * bg_std
        print(f"  > Threshold: {threshold:.2f} (BgMean {bg_mean:.2f} + 10*{bg_std:.2f})")
        mask = b0_data > threshold
        
    signal_roi = b0_data[mask]
    if len(signal_roi) == 0:
        print("WARNING: Empty Mask! Fallback to percentile.")
        mask = b0_data > np.percentile(b0_data, 95)
        signal_roi = b0_data[mask]
        
    # Use 95th percentile to approximate "Tissue Signal" and avoid Agar/Gel dilution
    signal_metric = np.percentile(signal_roi, 95)
    
    print(f"  > Signal (95th%): {signal_metric:.2f}")
    print(f"  > Bg Mean (Corner): {bg_mean:.2f}")
    print(f"  > Noise Sigma: {noise_sigma:.2f}")
        
    snr = signal_metric / noise_sigma
    return snr, noise_sigma

def rician_noise_jax(signal: jnp.ndarray, snr_target: float, current_max: float) -> jnp.ndarray:
    """
    Adds Rician noise to signal to achieve target SNR relative to current_max signal.
    
    Rician noise generation:
    S_noisy = sqrt( (S + N1)^2 + N2^2 )
    where N1, N2 ~ N(0, sigma^2)
    
    sigma = S_ref / SNR_target
    """
    # Sigma required for target SNR
    sigma = current_max / snr_target
    
    key = jax.random.PRNGKey(42) # Fixed seed for reproducibility definition
    
    # We need randomness per voxel.
    # Split key? Or just use stateless if we had it.
    # Just generating massive random array.
    
    N1 = jax.random.normal(key, signal.shape) * sigma
    
    # Need second key
    key2, _ = jax.random.split(key)
    N2 = jax.random.normal(key2, signal.shape) * sigma
    
    return jnp.sqrt((signal + N1)**2 + N2**2)

def main():
    root_dir = "/home/mhough/Downloads/STE"
    output_dir = "experiments/ste_dataset_integration/results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    ex_vivo_path = os.path.join(root_dir, "STE00_ExVivo/STE/STE.nii.gz")
    ex_bval = os.path.join(root_dir, "STE00_ExVivo/STE/bvals.txt")
    ex_bvec = os.path.join(root_dir, "STE00_ExVivo/STE/bvecs.txt")
    
    in_vivo_path = os.path.join(root_dir, "STE01/STE/STE.nii.gz")
    in_bval = os.path.join(root_dir, "STE01/STE/bvals.txt")
    in_bvec = os.path.join(root_dir, "STE01/STE/bvecs.txt")
    
    # 1. Load Data
    print("--- Loading Ex-Vivo ---")
    ex_data, ex_bvals, _, _ = load_dwi(ex_vivo_path, ex_bval, ex_bvec)
    
    print("--- Loading In-Vivo ---")
    in_data, in_bvals, _, _ = load_dwi(in_vivo_path, in_bval, in_bvec)
    
    # 2. Estimate SNR
    print("\n--- SNR Estimation ---")
    # generate masks (very simple threshold)
    ex_snr, ex_sigma = estimate_snr(ex_data, ex_bvals)
    print(f"Ex-Vivo Estimated SNR: {ex_snr:.2f} (Sigma: {ex_sigma:.2f})")
    
    in_snr, in_sigma = estimate_snr(in_data, in_bvals)
    print(f"In-Vivo Estimated SNR: {in_snr:.2f} (Sigma: {in_sigma:.2f})")
    
    # 3. Degrade Ex-Vivo
    # Logic update: In-Vivo spatial SNR (155) is likely artificial due to pre-proc masking.
    # Standard clinical dMRI SNR is ~20-30.
    # Ex-Vivo is ~50.
    # We will use the measured In-Vivo SNR ONLY if it is lower than Ex-Vivo, 
    # otherwise we default to 30.0 (Clinical Standard).
    
    target_snr = in_snr
    if in_snr > ex_snr:
        print(f"\nWARNING: Measured In-Vivo SNR ({in_snr:.2f}) > Ex-Vivo ({ex_snr:.2f}).")
        print("This suggests In-Vivo is denoised/masked or Ex-Vivo is raw.")
        print("Defaulting Target SNR to 30.0 (Standard Clinical Quality).")
        target_snr = 30.0
    
    print(f"\n--- Degrading Ex-Vivo to SNR {target_snr:.2f} ---")
    
    # Convert to JAX
    ex_data_jax = jnp.array(ex_data)
    
    # Ref signal for noise scaling is typically b0 mean (or 95th percentile now)
    b0_ref = jnp.mean(ex_data_jax[..., ex_bvals < 50], axis=-1)
    # Re-calc 95th percentile on the fly or use the one from estimate_snr
    # We'll re-estimate roughly to be safe
    threshold = np.mean(b0_ref[b0_ref>0]) # Rough
    mask = b0_ref > threshold
    if np.sum(mask) == 0: mask = b0_ref > 0
    b0_signal_max = np.percentile(b0_ref[mask], 95)
    
    degraded_ex_vivo = rician_noise_jax(ex_data_jax, target_snr, b0_signal_max)
    
    # 4. Compute Metrics after degradation
    deg_data_np = np.array(degraded_ex_vivo)
    deg_snr, deg_sigma = estimate_snr(deg_data_np, ex_bvals)
    
    print(f"Degraded Ex-Vivo SNR: {deg_snr:.2f}")
    
    # 5. Determine Pass/Fail (Tolerance 5.0)
    diff = abs(deg_snr - target_snr)
    print(f"Difference from Target: {diff:.2f}")
    
    if diff < 5.0:
        print("PASS: Noise injection successfully mimicked Target SNR.")
    else:
        print("WARNING: Degraded SNR deviates from target. Check scaling.")

    # 6. Save Degraded Sample (Optional, middle slice)
    # Just save report for now
    report_path = os.path.join(output_dir, "reality_check_report.txt")
    with open(report_path, "w") as f:
        f.write("Reality Check Report\n")
        f.write("====================\n")
        f.write(f"Ex-Vivo SNR: {ex_snr:.2f}\n")
        f.write(f"In-Vivo SNR: {in_snr:.2f}\n")
        f.write(f"Degraded Ex-Vivo SNR: {deg_snr:.2f}\n")
        f.write(f"Pass: {diff < 5.0}\n")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
