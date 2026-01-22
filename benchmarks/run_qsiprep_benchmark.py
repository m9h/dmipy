import os
import sys
import numpy as np
import nibabel as nib
import scipy.ndimage
from scipy.ndimage import gaussian_gradient_magnitude

# Helper to find adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adapters.swinir_adapter import SwinIRPredictor

def compute_gradient_correlation(img_data, ref_data):
    """
    Computes Gradient Correlation between two volumes.
    """
    # Normalize percentiles to handle intensity scaling differences
    img_norm = img_data / (np.percentile(img_data, 99) + 1e-6)
    ref_norm = ref_data / (np.percentile(ref_data, 99) + 1e-6)
    
    grad_img = gaussian_gradient_magnitude(img_norm, sigma=1.0)
    grad_ref = gaussian_gradient_magnitude(ref_norm, sigma=1.0)
    
    # Mask out background (robustness)
    # T1 usually has clear background
    mask = (ref_norm > 0.05) & (grad_ref > 0)
    
    g1 = grad_img[mask].flatten()
    g2 = grad_ref[mask].flatten()
    
    if len(g1) == 0 or np.std(g1) == 0 or np.std(g2) == 0:
        return 0.0
        
    corr = np.corrcoef(g1, g2)[0, 1]
    return 0.0 if np.isnan(corr) else corr

def resample_to_ref(img_data, img_affine, ref_data, ref_affine):
    """
    Simple affine resampling using scipy.ndimage.affine_transform.
    Only handles scale/translation roughly or assumes close alignment.
    For robust resampling we usually use nilearn, but we want to minimize heavy deps if possible.
    Let's use nilearn if installed, else fallback to zoom if affines are diagonal.
    """
    try:
        from nilearn.image import resample_img
        # Create Nifti images
        src_img = nib.Nifti1Image(img_data, img_affine)
        dst_img = nib.Nifti1Image(ref_data, ref_affine)
        
        # Resample
        resampled = resample_img(src_img, target_affine=ref_affine, target_shape=ref_data.shape, interpolation='linear')
        return resampled.get_fdata()
    except ImportError:
        print("Warning: nilearn not found. Falling back to scipy zoom (assuming aligned inputs).")
        # Start with simple zoom based on shape ratio
        zoom_factors = np.array(ref_data.shape) / np.array(img_data.shape)
        # 3D only
        if img_data.ndim > 3:
             zoom_factors = np.append(zoom_factors, [1]*(img_data.ndim-3))
        
        return scipy.ndimage.zoom(img_data, zoom_factors, order=1)

def run_benchmark():
    # Define Subjects
    # Note: Using absolute paths derived from previous browsing
    base_data = "data/ds004910/derivatives"
    
    subjects = [
        {
            "id": "sub-01",
            "dwi": f"{base_data}/dwipreproc/sub-01/ses-02/dwi/sub-01_ses-02_dwi.nii.gz",
            "t1": f"{base_data}/anatpreproc/sub-01/anat/sub-01_ses-01_T1w.nii.gz"
        },
        {
            "id": "sub-03",
            # Assuming ses-01 for sub-03 exists based on previous `find` (step 150 showed ses-01/dwi... wait, step 150 showed dwipreproc/sub-03/ses-01/dwi/...)
            # Actually, let's verify if ses-01 has the dwi.
            # Step 150 output: ds004910/derivatives/dwipreproc/sub-03/ses-01/dwi/sub-03_ses-01_dwi.nii.gz
            "dwi": f"{base_data}/dwipreproc/sub-03/ses-01/dwi/sub-03_ses-01_dwi.nii.gz",
            "t1": f"{base_data}/anatpreproc/sub-03/anat/sub-03_ses-01_T1w.nii.gz"
        }
    ]
    
    # Initialize SwinIR
    # Scale 4 is standard SwinIR.
    scale = 4
    predictor = SwinIRPredictor(scale=scale, device='cuda') # Use CUDA if available
    
    print(f"{'Subject':<10} | {'Method':<20} | {'GradCorrelation':<15}")
    print(f"CWD: {os.getcwd()}")
    print("-" * 55)
    
    for subj in subjects:
        print(f"Checking {subj['id']} path: {os.path.abspath(subj['dwi'])}")
        if not os.path.exists(subj["dwi"]):
            print(f"Skipping {subj['id']}: DWI not found {subj['dwi']}")
            continue
            
        # Load Data
        dwi_img = nib.load(subj["dwi"])
        t1_img = nib.load(subj["t1"])
        
        dwi_data = dwi_img.get_fdata()
        t1_data = t1_img.get_fdata()
        
        # Take first volume for speed/relevance (b0 or first encoding)
        # Gradient Correl usually on b0 or mean b0 or high contrast vol
        if dwi_data.ndim == 4:
            src_vol = dwi_data[..., 0] 
        else:
            src_vol = dwi_data
            
        # --- Method 1: Baseline (Interpolation) ---
        # Resample native DWI to T1 Grid directly
        # Sinc interpolation (approximated by order=3 splines or nilearn 'linear' for speed if acceptable, 
        # but SwinIR comparison should be vs better interp. Let's use order=3)
        
        # We need resample_to_ref with order=3 logic
        # For simplicity in this script using nilearn wrapper above:
        # Note: nilearn resample_img uses interpolation='continuous' (linear) or 'nearest' or 'linear'. 
        # Actually nilearn uses scipy.ndimage.affine_transform.
        # Let's trust nilearn linear for baseline "Standard Interpolation".
        baseline_resampled = resample_to_ref(src_vol, dwi_img.affine, t1_data, t1_img.affine)
        
        score_base = compute_gradient_correlation(baseline_resampled, t1_data)
        print(f"{subj['id']:<10} | {'Baseline (Linear)':<20} | {score_base:.4f}")
        
        # --- Method 2: SwinIR ---
        # 1. Run inference (Low Res -> High Res (x4))
        # Need to reshape 2D->4D for volume? My adapter takes 4D (H,W,Z,1)
        sr_input = src_vol[..., np.newaxis] # (H,W,Z,1)
        sr_output = predictor.predict_volume(sr_input) # (H*4, W*4, Z, 1)
        sr_output = sr_output[..., 0] # (H*4, W*4, Z)
        
        # 2. Construct Affine for SR Output
        # Assuming simple scaling of first 2 dims
        sr_affine = dwi_img.affine.copy()
        sr_affine[0, 0] /= scale
        sr_affine[1, 1] /= scale
        
        # 3. Resample SR (High Res) to T1 Grid (Target High Res)
        # This step handles misalignment or slight FOV differences
        swinir_resampled = resample_to_ref(sr_output, sr_affine, t1_data, t1_img.affine)
        
        score_swin = compute_gradient_correlation(swinir_resampled, t1_data)
        print(f"{subj['id']:<10} | {'SwinIR (x4)':<20} | {score_swin:.4f}")

if __name__ == "__main__":
    run_benchmark()
