
import os
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import nibabel as nib
import jax.numpy as jnp
from dmipy_jax.data.openneuro import DEFAULT_DATA_DIR

# Dataset ID: ds006557 (Ultra-low-field brain MRI)
DS006557_BUCKET = "openneuro.org"
DS006557_PREFIX = "ds006557"

def fetch_ds006557(path: Path = None, subject: str = "sub-01", download: bool = True) -> Path:
    """
    Fetches the Ultra-low-field brain MRI dataset (ds006557) from OpenNeuro via S3.
    Downloads only the specified subject.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / "ds006557"
        
    if not download:
        return path

    path.mkdir(parents=True, exist_ok=True)
    
    # Setup S3 client for anonymous access
    s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
    
    # Try both sub-01 and sub-001 patterns if not sure, but default calls use strict subject
    # OpenNeuro usually uses sub-XX.
    # Updated: ds006557 is versioned. Use version 1.0.2 path.
    version = "1.0.2"
    prefix = f"{DS006557_PREFIX}/versions/{version}/{subject}/"
    
    # Fallback to invalidating version if empty? No, typically stick to version.
    print(f"Fetching {subject} from s3://{DS006557_BUCKET}/{prefix}")
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=DS006557_BUCKET, Prefix=prefix)

    found_any = False
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            found_any = True
            key = obj['Key']
            # Remove dataset prefix for local path (strip version part for cleaner local structure)
            # key: ds006557/versions/1.0.2/sub-01/
            # local: ds006557/sub-01/
            rel_path = key.replace(f"{DS006557_PREFIX}/versions/{version}/", "", 1)
            local_file = path / rel_path
            
            if local_file.exists():
                continue
                
            print(f"Downloading {key} -> {local_file}")
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(DS006557_BUCKET, key, str(local_file))
            
    if not found_any:
        # Fallback to root if version fails?
        print(f"WARNING: No objects found for version {version}. Trying root...")
        prefix_root = f"{DS006557_PREFIX}/{subject}/"
        pages = paginator.paginate(Bucket=DS006557_BUCKET, Prefix=prefix_root)
        for page in pages:
             if 'Contents' in page:
                 found_any = True
                 # ... download logic ...
                 print("Found in root (not implemented fully in this snippet, relying on version)")
                 break
        if not found_any:
             print(f"ERROR: Subject {subject} not found in {version} or root.")

            
    return path

def load_ds006557_data(path: Path = None, subject: str = "sub-01", contrast: str = "T1w"):
    """
    Loads ds006557 MRI data.
    
    Args:
        path: Path to dataset root.
        subject: Subject ID.
        contrast: 'T1w' or 'T2w' (or others available).
    
    Returns:
        Dictionary containing image data and affine.
    """
    if path is None:
        path = fetch_ds006557(subject=subject, download=True)
        
    # Construct path: sub-001/anat/sub-001_T1w.nii.gz
    # Note: Filenames might vary slightly (e.g. acq sequences), let's glob.
    subj_dir = path / subject / "anat"
    
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject anatomy directory not found: {subj_dir}")
        
    pattern = f"*{contrast}.nii.gz"
    files = list(subj_dir.glob(pattern))
    
    if not files:
        # Try finding json to see what's available?
        # Or maybe it's under ses-xx?
        # Check for session directories
        sessions = list(path.glob(f"{subject}/ses-*"))
        if sessions:
             # Try first session
             subj_dir = sessions[0] / "anat"
             files = list(subj_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No {contrast} NIfTI found in {subj_dir}")
        
    # Pick the first one (often there's just one, or maybe 'acq-lowfield')
    # Use the one with 'lowfield' in name if available?
    # The dataset title says "Ultra-low-field", so maybe they are all low field?
    # But description says "correspondence to high-field".
    # We should distinguish.
    
    target_file = files[0]
    
    # Try to prefer 'low' field if multiple
    for f in files:
        if "low" in f.name.lower():
            target_file = f
            break
            
    print(f"Loading {target_file}")
    img = nib.load(target_file)
    data = img.get_fdata()
    affine = img.affine
    
    return {
        'image': jnp.array(data),
        'affine': affine,
        'path': str(target_file)
    }
