
import os
from pathlib import Path
import nibabel as nib
import jax.numpy as jnp

# Default location for MNE Sample Data
MNE_DATA_PATH = Path.home() / "mne_data" / "MNE-sample-data"

def load_mne_sample_data(path: Path = None):
    """
    Loads the MNE Sample Dataset MRI (T1-weighted).
    This dataset typically comes with MNE-Python.
    
    Args:
        path: Path to MNE-sample-data root. Defaults to ~/mne_data/MNE-sample-data.
        
    Returns:
        Dictionary containing image data and affine.
    """
    if path is None:
        path = MNE_DATA_PATH
        
    if not path.exists():
        raise FileNotFoundError(f"MNE Sample Data not found at {path}. Please install mne-python or download the sample data.")
        
    # Path: subjects/sample/mri/T1.mgz
    # We use nibabel which supports .mgz (FreeSurfer format)
    t1_path = path / "subjects" / "sample" / "mri" / "T1.mgz"
    
    if not t1_path.exists():
        # Fallback? Maybe .nii?
        t1_path_nii = path / "subjects" / "sample" / "mri" / "T1.nii"
        if t1_path_nii.exists():
            t1_path = t1_path_nii
        else:
             t1_path_nii_gz = path / "subjects" / "sample" / "mri" / "T1.nii.gz"
             if t1_path_nii_gz.exists():
                 t1_path = t1_path_nii_gz
             else:
                raise FileNotFoundError(f"T1 volume not found in {path}/subjects/sample/mri/")
                
    print(f"Loading MNE Sample T1 from {t1_path}")
    img = nib.load(t1_path)
    data = img.get_fdata()
    affine = img.affine
    
    return {
        'image': jnp.array(data),
        'affine': affine,
        'path': str(t1_path)
    }
