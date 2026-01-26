import nibabel as nib
import numpy as np
import glob
import os

print(f"{'File':<40} | {'Mean':<10} | {'Min':<10} | {'Max':<10}")
print("-" * 80)

files = sorted(glob.glob('results/**/*.nii.gz', recursive=True))

for f in files:
    try:
        img = nib.load(f)
        data = img.get_fdata()
        # Filter Background
        mask = data > 0
        if np.sum(mask) == 0:
            print(f"{os.path.basename(f):<40} | {'EMPTY':<10} | {'-':<10} | {'-':<10}")
            continue
            
        vals = data[mask]
        print(f"{os.path.basename(f):<40} | {np.mean(vals):<10.4f} | {np.min(vals):<10.4f} | {np.max(vals):<10.4f}")
    except Exception as e:
        print(f"Error loading {f}: {e}")
