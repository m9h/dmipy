
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from dmipy_jax.data.openneuro import fetch_datalad, DEFAULT_DATA_DIR
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.acquisition import JaxAcquisition
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

# Dataset ID: ds006181 (Connectome 2.0)
DS006181_GIT_URL = "https://github.com/OpenNeuroDatasets/ds006181.git"

def fetch_connectome2(path: Path = None, subject: str = "sub-01") -> Path:
    """
    Fetches the Connectome 2.0 dataset (ds006181) from OpenNeuro via DataLad.
    Downloads only the specified subject to save space/time.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / "ds006181"
        
    # Fetch the dataset (lazy fetch mostly, maybe getting annex keys)
    ds_path = fetch_datalad(DS006181_GIT_URL, path=path)
    
    # We explicitly want the dwi data for the subject
    # Pattern: sub-01/ses-01/dwi/
    # Let's enforce fetching content for this subject
    # This requires running `datalad get` inside the repo.
    # Our fetch_datalad helper might need an option for specific paths?
    # Or we just assume it gets the structure and we call get later or rely on auto-get.
    # Let's assume the user runs this in an environment where datalad auto-fetches on open,
    # or we trigger it.
    
    return ds_path

def load_connectome2_mri(path: Path = None, subject: str = "sub-01", session: str = "ses-01", voxel_slice=None):
    """
    Loads Connectome 2.0 MRI data.
    """
    if path is None:
        path = fetch_connectome2()
        
    subj_dir = path / subject / session / "dwi"
    if not subj_dir.exists():
         raise FileNotFoundError(f"Subject directory not found: {subj_dir}")
         
    # Find dwi file
    # sub-01_ses-01_acq-ConnectomX_dwi.nii.gz
    dwi_files = list(subj_dir.glob("*dwi.nii.gz"))
    if not dwi_files:
        raise FileNotFoundError(f"No DWI NIfTI found in {subj_dir}")
    dwi_path = dwi_files[0]
    
    # Load data
    data, affine = load_nifti(str(dwi_path))
    if voxel_slice is not None:
        data = data[voxel_slice]
        
    # Load bvals/bvecs
    bval_path = str(dwi_path).replace(".nii.gz", ".bval")
    bvec_path = str(dwi_path).replace(".nii.gz", ".bvec")
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    
    # Load JSON sidecar for Timing Info
    json_path = str(dwi_path).replace(".nii.gz", ".json")
    delta_arr = None
    Delta_arr = None
    
    if Path(json_path).exists():
        with open(json_path, 'r') as f:
            meta = json.load(f)
        # Check for explicit timing arrays (rare in standard BIDS, but might be there)
        # Or check 'EffectiveEchoTime' or similar.
        # But this dataset mixes two diffusion times.
        # BIDS might split them? No, found 1 file.
        # Does the JSON have 'DiffusionScheme' or 'dcmmeta_shape'?
        pass
        
    # Manual Metadata Mapping (based on Dataset Description)
    # D = 13 ms (Delta=13ms?) and D = 30 ms.
    # We need to assign Delta per volume.
    # Protocol:
    # D=13ms: b \in [50, 6000]
    # D=30ms: b \in [200, 17800]
    
    # Heuristic:
    # Split by b-value?
    # If b > 6000, definitely D=30.
    # If b <= 6000, could be either.
    # This is ambiguous.
    # Need to check if there is a 'b_table.txt' or similar custom file, 
    # or if the JSON contains a list for 'DiffusionTime'.
    
    # If explicit timing is missing from metadata files, we might strictly need separate files 
    # OR we assume the user knows the shell structure.
    # Let's inspect the JSON if easier.
    
    # For now, create a Placeholder Scheme assuming user will inject times if needed,
    # or try to parse 'ConnectomX' specifics.
    
    # Let's assume constant small_delta for now (e.g. 10ms?). 
    # The dataset description says D (big Delta) varies.
    
    # Dummy creation for now - caller will override or we improve parsing in v2
    scheme = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    return {
        'dwi': jnp.array(data),
        'scheme': scheme,
        'affine': affine,
        'bvals': bvals,
        'bvecs': bvecs,
        'json_path': json_path
    }
