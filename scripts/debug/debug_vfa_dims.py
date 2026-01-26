
import nibabel as nib
import os

base = "/home/mhough/dev/dmipy/data/wand/sub-00395/ses-02/anat"
files = [
    "sub-00395_ses-02_acq-spgr_part-mag_VFA.nii.gz",
    "sub-00395_ses-02_acq-ssfp_part-mag_VFA.nii.gz",
    "sub-00395_ses-02_acq-spgrIR_part-mag_VFA.nii.gz"
]

for f in files:
    path = os.path.join(base, f)
    if os.path.exists(path):
        img = nib.load(path)
        print(f"File: {f}")
        print(f"  Shape: {img.shape}")
        # print header check?
    else:
        print(f"File not found: {f}")
