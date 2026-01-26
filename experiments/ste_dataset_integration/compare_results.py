import nibabel as nib
import numpy as np
import os
import glob

def compute_stats(path, name):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
        
    img = nib.load(path)
    data = img.get_fdata()
    
    # Mask zeros (background)
    mask = data > 0
    if np.sum(mask) == 0:
        return 0, 0
        
    vals = data[mask]
    
    # Robust stats (median, IQR)
    median = np.median(vals)
    mean = np.mean(vals)
    std = np.std(vals)
    
    print(f"{name}: Mean={mean:.4e}, Std={std:.4e}, Median={median:.4e}")
    return mean, std

def main():
    print("--- Comparative Analysis: In-Vivo vs Ex-Vivo ---")
    
    base_dir = "experiments/ste_dataset_integration/results"
    
    metrics = ['D_intra', 'D_extra', 'f_intra', 'exchange_time']
    
    results = {}
    
    for metric in metrics:
        print(f"\nMetric: {metric}")
        ex_path = f"{base_dir}/ex_vivo_{metric}.nii.gz"
        in_path = f"{base_dir}/in_vivo_{metric}.nii.gz"
        
        print("  [Ex-Vivo]")
        stats_ex = compute_stats(ex_path, "ExVivo")
        
        print("  [In-Vivo]")
        stats_in = compute_stats(in_path, "InVivo")
        
        if stats_ex and stats_in:
            diff = stats_ex[0] - stats_in[0]
            print(f"  Difference (Ex - In): {diff:.4e}")
            
if __name__ == "__main__":
    main()
