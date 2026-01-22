import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

def compute_gradient_correlation(img_data, ref_data):
    # Compute Grad Magnitude
    # Normalize
    img_data = img_data / (np.percentile(img_data, 99) + 1e-6)
    ref_data = ref_data / (np.percentile(ref_data, 99) + 1e-6)
    
    grad_img = gaussian_gradient_magnitude(img_data, sigma=1.0)
    grad_ref = gaussian_gradient_magnitude(ref_data, sigma=1.0)
    
    # Mask out background
    mask = (ref_data > 0.1) & (grad_img > 0) & (grad_ref > 0)
    
    g1 = grad_img[mask].flatten()
    g2 = grad_ref[mask].flatten()
    
    if len(g1) == 0 or np.std(g1) == 0 or np.std(g2) == 0:
        return 0.0
        
    corr = np.corrcoef(g1, g2)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr

def main():
    sub = "sub-01"
    base_dir = f"/home/mhough/datasets/ds001957-study/derivatives/super_resolution/{sub}/dwi"
    t1_path = f"/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/{sub}/anat/t1.nii.gz"
    
    methods = {
        "Baseline (Sinc)": "sub-01_desc-baseline_dwi.nii.gz",
        "MMORF": "sub-01_desc-mmorf_dwi.nii.gz",
        "JTV": "sub-01_desc-jtv_dwi.nii.gz",
        "INR": "sub-01_desc-inr_dwi.nii.gz"
    }
    
    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata()
    
    print(f"--- Super Resolution Metrics (Gradient Correlation with T1) ---")
    
    import json
    results = {}
    
    for name, fname in methods.items():
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"{name}: File not found ({fpath})")
            results[name] = "N/A"
            continue
            
        img = nib.load(fpath)
        data = img.get_fdata()
        
        # If 4D, take first volume
        if data.ndim == 4:
            data = data[..., 0]
            
        # Ensure shape match
        if data.shape != t1_data.shape:
            # Resample? Or just skip/warn. 
            # Our pipelines should have produced T1 grid.
            print(f"{name}: Shape mismatch {data.shape} vs {t1_data.shape}")
            results[name] = "Shape Mismatch"
            continue
            
        score = compute_gradient_correlation(data, t1_data)
        print(f"{name}: {score:.4f}")
        results[name] = score

    # Save to JSON
    out_json = os.path.join(base_dir, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to {out_json}")

if __name__ == "__main__":
    main()
