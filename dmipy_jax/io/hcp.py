"""
HCP Data Loader using Boto3 (S3).
Requires AWS credentials with access to the hcp-openaccess bucket.
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import nibabel as nib
import jax.numpy as jnp
import numpy as np
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues

BUCKET_NAME = 'hcp-openaccess'

def get_hcp_client():
    """
    Returns a boto3 s3 client.
    If credentials are not found, it might attempt anonymous access, 
    but HCP usually requires authenticated keys.
    """
    return boto3.client('s3')

def download_hcp_subject(subject_id: str, output_dir: str, overwrite: bool = False):
    """
    Downloads structural and diffusion data for a given HCP subject.
    
    Files downloaded:
    - T1w/T1w_acpc_dc_restore_1.25.nii.gz
    - T1w/Diffusion/data.nii.gz
    - T1w/Diffusion/bvals
    - T1w/Diffusion/bvecs
    """
    s3 = get_hcp_client()
    prefix = f"HCP_1200/{subject_id}"
    
    files_to_fetch = {
        "T1w": f"{prefix}/T1w/T1w_acpc_dc_restore_1.25.nii.gz",
        "dMRI": f"{prefix}/T1w/Diffusion/data.nii.gz",
        "bvals": f"{prefix}/T1w/Diffusion/bvals",
        "bvecs": f"{prefix}/T1w/Diffusion/bvecs"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    local_paths = {}
    
    for key, s3_key in files_to_fetch.items():
        filename = os.path.basename(s3_key)
        local_path = os.path.join(output_dir, filename)
        local_paths[key] = local_path
        
        if os.path.exists(local_path) and not overwrite:
            print(f"Skipping {filename} (exists).")
            continue
            
        print(f"Downloading {s3_key} to {local_path}...")
        try:
            s3.download_file(BUCKET_NAME, s3_key, local_path)
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            if key == "T1w": # Critical failure
                raise
                
    return local_paths

def load_hcp_subject(subject_id: str, data_root: str, download: bool = True) -> tuple:
    """
    Loads HCP subject data. Downloads if not present.
    
    Returns:
        (data_dMRI, scheme, data_T1)
    """
    subject_dir = os.path.join(data_root, subject_id)
    
    if download:
        paths = download_hcp_subject(subject_id, subject_dir)
    else:
        # Assume paths
        paths = {
            "T1w": os.path.join(subject_dir, "T1w_acpc_dc_restore_1.25.nii.gz"),
            "dMRI": os.path.join(subject_dir, "data.nii.gz"),
            "bvals": os.path.join(subject_dir, "bvals"),
            "bvecs": os.path.join(subject_dir, "bvecs")
        }
    
    # Load NIfTI
    print("Loading NIfTI files...")
    img_dwi = nib.load(paths['dMRI'])
    data_dwi = jnp.array(img_dwi.get_fdata())
    
    img_t1 = nib.load(paths['T1w'])
    data_t1 = jnp.array(img_t1.get_fdata())
    
    # Scheme
    from dipy.io.gradients import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(paths['bvals'], paths['bvecs'])
    
    # Standardize bvals to SI (s/m^2)
    # HCP bvals are s/mm^2 (0, 1000, 2000, 3000)
    bvals_si = bvals * 1e6
    scheme = acquisition_scheme_from_bvalues(bvals_si, bvecs, delta=0.01, Delta=0.03, TE=0.07)
    
    return data_dwi, scheme, data_t1
