import nibabel as nib
import numpy as np
import os

sub = "sub-NDARAA306NT2"
root = "data/hbn_aws"
dwi_path = f"{root}/{sub}/dwi/{sub}_acq-64dir_dwi.nii.gz"
bval_path = f"{root}/{sub}/dwi/{sub}_acq-64dir_dwi.bval"

if not os.path.exists(dwi_path):
    # Try finding it via glob like the script
    import glob
    g = glob.glob(f"{root}/{sub}/*dwi.nii.gz")
    if g: dwi_path = g[0]
    
print(f"Loading {dwi_path}")
img = nib.load(dwi_path)
data = img.get_fdata()

bvals = np.loadtxt(bval_path)
print(f"B-values: {bvals[:10]} ... {bvals[-5:]}")
print(f"B0 count: {np.sum(bvals < 50)}")
print(f"Data Shape: {data.shape}")
print(f"Data Max: {np.max(data)}")
print(f"Data Min: {np.min(data)}")
print(f"Data Mean: {np.mean(data)}")

# Check center voxel
center = np.array(data.shape[:3]) // 2
print(f"Center Voxel Signal: {data[center[0], center[1], center[2]]}")

# Normalization Check
b0_mask = bvals < 50
if np.sum(b0_mask) > 0:
    S0 = np.mean(data[..., b0_mask], axis=-1)
    print(f"S0 Max: {np.max(S0)}")
    print(f"S0 Min: {np.min(S0)}")
    
    # Check for zeros in S0
    zeros = np.sum(S0 == 0)
    print(f"S0 Zeros: {zeros}")
