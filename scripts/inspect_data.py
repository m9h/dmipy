
import nibabel as nib
import numpy as np
from pathlib import Path

base = Path("/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep")
subjects = ["sub-01", "sub-03"]

print("--- Inspecting Input Data ---")
for sub in subjects:
    t1_path = base / sub / "anat" / "t1.nii.gz"
    mask_path = base / sub / "anat" / "mask.nii.gz"
    
    if t1_path.exists():
        img = nib.load(t1_path)
        print(f"\nSubject: {sub}")
        print(f"Path: {t1_path}")
        print(f"Shape: {img.shape}")
        print(f"Affine:\n{img.affine}")
        
        # Check center of mass or simple intensity center
        data = img.get_fdata()
        center_of_mass = np.array(np.where(data > data.mean())).mean(axis=1)
        print(f"Data Center of Mass (indices): {center_of_mass}")
        
        # Check hardcoded crop region [40:200, 12:236, 80:240] (from transorms.py, assuming LAS orientation)
        # Note: transorms.py uses SpatialCropd with roi_start/end.
        # We need to know if this ROI makes sense for this data.
        # But wait, MONAI transforms apply AFTER Orientationd(LAS). 
        # We don't know the original orientation without checking the affine (which we are doing).
        
    if mask_path.exists():
        mask = nib.load(mask_path)
        print(f"Mask Shape: {mask.shape}")
        mdata = mask.get_fdata()
        mask_center = np.array(np.where(mdata > 0)).mean(axis=1)
        print(f"Mask Center (indices): {mask_center}")
