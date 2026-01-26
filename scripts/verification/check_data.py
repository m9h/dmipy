
import nibabel as nib
import numpy as np
import sys

try:
    img = nib.load('data/ds003563/sub-yv98/ses-4291/dwi/sub-yv98_ses-4291_dwi.nii.gz')
    print(f"Shape: {img.shape}")
    print(f"Affine diagonal: {np.diag(img.affine)[:3]}")
    print(f"Voxel Size: {img.header.get_zooms()[:3]}")
    
    bvals = np.loadtxt('data/ds003563/sub-yv98/ses-4291/dwi/sub-yv98_ses-4291_dwi.bval')
    print(f"B-values: {bvals}")
except Exception as e:
    print(f"Error: {e}")
