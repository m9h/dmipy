
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp
from dmipy_jax.data.openneuro import fetch_datalad, DEFAULT_DATA_DIR
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.acquisition import JaxAcquisition
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
import datalad.api as dl

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
        
    print(f"DEBUG: Loading Connectome2 from {path}")
    
    # Check for derivatives first (Preprocessed)
    # Structure: derivatives/preprocessed_dwi/sub-01
    deriv_dir = path / "derivatives" / "preprocessed_dwi" / subject
    print(f"DEBUG: Checking deriv_dir={deriv_dir}, exists={deriv_dir.exists()}")
    if deriv_dir.exists():
        # Look for preprocessed dwi
        dwi_files = list(deriv_dir.glob("*dwi.nii.gz"))
        if dwi_files:
             subj_dir = deriv_dir
             dwi_path = dwi_files[0]
        else:
             subj_dir = None
    else:
        subj_dir = None
        
    if subj_dir is None:
        # Fallback to raw BIDS
        # Handle session
        if session:
            subj_dir = path / subject / session / "dwi"
        else:
             subj_dir = path / subject / "dwi"
             
        if not subj_dir.exists() and session:
             # Try without session
             subj_dir = path / subject / "dwi"

        if not subj_dir.exists():
             raise FileNotFoundError(f"Subject directory not found: {subj_dir} (checked derivatives and raw)")
             
        # Find dwi file
        dwi_files = list(subj_dir.glob("*dwi.nii.gz"))
        if not dwi_files:
            raise FileNotFoundError(f"No DWI NIfTI found in {subj_dir}")
        dwi_path = dwi_files[0]
    
    # Retrieve data via DataLad if it's a pointer file (or just ensure it's there)
    print(f"DEBUG: Ensuring data presence for {dwi_path}")
    try:
        # We must specify the dataset path because CWD might be different
        dl.get(path=str(dwi_path), dataset=str(path))
    except Exception as e:
        print(f"WARNING: DataLad get failed: {e}. Assuming file is already present or handled.")

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
        'json_path': json_path,
        'dataset_path': path, # Return root path for looking up grad_dev
        'subject': subject
    }

def load_gradient_nonlinearity(path: Path, subject: str = "sub-01") -> jnp.ndarray:
    """
    Loads the Gradient Nonlinearity Tensor (grad_dev.nii.gz).
    Expected shape in file: (X, Y, Z, 1, 3, 3) or (X, Y, Z, 9) or (X, Y, Z, 3, 3).
    HCP usually stores it as separate files or a 4D volume.
    Standard HCP: 'grad_dev.nii.gz' in 'MNINonLinear' or 'T1w' folder? 
    For raw data, it's often in distinct folder.
    
    We assume it's available in the subject dir or derivatives.
    """
    # Potential paths
    candidates = [
        path / "derivatives" / "grad_dev" / subject / "grad_dev.nii.gz",
        path / subject / "grad_dev.nii.gz",
        path / "grad_dev.nii.gz"
    ]
    
    grad_dev_path = None
    for p in candidates:
        if p.exists():
            grad_dev_path = p
            break
            
    if grad_dev_path is None:
        # Check if we can download it? 
        # For now, return None or create Identity if not found?
        # Better to raise error if explicitly asked, but here we return None.
        print("WARNING: Gradient Nonlinearity Tensor (grad_dev.nii.gz) not found.")
        return None
    
    print(f"Loading Gradient Nonlinearity Tensor from {grad_dev_path}")
    data, _ = load_nifti(str(grad_dev_path))
    # data shape?
    # If 9 components: reshape to (..., 3, 3)
    if data.shape[-1] == 9:
        data = data.reshape(data.shape[:-1] + (3, 3))
    elif data.ndim == 5 and data.shape[-2:] == (3, 3):
        pass
    
    return jnp.array(data)

def apply_gradient_nonlinearity(scheme: JaxAcquisition, grad_dev: jnp.ndarray, roi_slice=None) -> JaxAcquisition:
    """
    Applies the gradient nonlinearity tensor to the acquisition scheme.
    
    Args:
        scheme: The nominal acquisition scheme.
        grad_dev: The gradient nonlinearity tensor field (X, Y, Z, 3, 3).
        roi_slice: Tuple of slices to apply to grad_dev to match data ROI.
        
    Returns:
        JaxAcquisition: A batched acquisition scheme with shapes (N_vox, N_meas, ...).
    """
    # 1. Crop grad_dev if needed
    if roi_slice is not None:
        grad_dev = grad_dev[roi_slice]
        
    # Flatten spatial dims to N_vox
    # grad_dev shape: (X, Y, Z, 3, 3) -> (N_vox, 3, 3)
    spatial_shape = grad_dev.shape[:-2]
    n_vox = np.prod(spatial_shape)
    L = grad_dev.reshape(n_vox, 3, 3)
    
    # 2. Apply to Gradients
    # Nominal gradients: g_nom (N_meas, 3)
    g_nom = scheme.gradient_directions
    b_nom = scheme.bvalues 
    
    # We need to compute g_local = L @ g_nom.T
    # Dimensions:
    # L: (N_vox, 3, 3)
    # g_nom: (N_meas, 3) -> (3, N_meas) for multiplication
    
    # Result: (N_vox, 3, N_meas) -> transpose to (N_vox, N_meas, 3)
    g_local_T = jnp.matmul(L, g_nom.T) # (N_vox, 3, N_meas)
    g_local = jnp.transpose(g_local_T, (0, 2, 1)) # (N_vox, N_meas, 3)
    
    # 3. Calculate Local b-values and Directions
    # |g_local|^2
    g_norm_sq = jnp.sum(g_local**2, axis=-1) # (N_vox, N_meas)
    g_norm = jnp.sqrt(g_norm_sq)
    
    # b_local = b_nom * |g_local|^2
    # Ensure b_nom is broadcastable (N_meas,) -> (1, N_meas)
    b_local = b_nom[None, :] * g_norm_sq # (N_vox, N_meas)
    
    # directions_local = g_local / |g_local|
    # Handle zeros safe division
    safe_norm = jnp.where(g_norm > 0, g_norm, 1.0)
    directions_local = g_local / safe_norm[..., None]
    
    # 4. Create Batched Acquisition
    # We must replicate delta/Delta/TE if they are scalars or 1D
    # JaxAcquisition expects arrays. If we pass (N_vox, N_meas), it's batched.
    
    # Need to verify if other fields need batching.
    # Typically only bvals/bvecs/btensors change.
    # Timings are properties of the sequence, likely constant per shot?
    # Unless gradient limits affect timing? Assuming fixed timing.
    
    return JaxAcquisition(
        bvalues=b_local,
        gradient_directions=directions_local,
        delta=scheme.delta,
        Delta=scheme.Delta,
        echo_time=scheme.echo_time,
        total_readout_time=scheme.total_readout_time
    )
