"""
Data loading utilities for dmipy-jax.
Provides access to standard validation datasets (Stanford, Sherbrooke) and
specific loaders for the BigMac dataset.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import jax.numpy as jnp
import numpy as np
import nibabel as nib
try:
    from dipy.data import fetch_stanford_hardi, read_stanford_hardi, fetch_sherbrooke_3shell, read_sherbrooke_3shell, get_fnames
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    DIPY_AVAILABLE = True
except ImportError:
    DIPY_AVAILABLE = False

from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.acquisition import JaxAcquisition

def load_stanford_hardi(voxel_slice=None) -> Tuple[jnp.ndarray, Any]:
    """
    Loads the Stanford HARDI dataset from DIPY.
    
    Args:
        voxel_slice: Optional tuple of slices to crop the data.
                     e.g. (slice(30, 40), slice(30, 40), slice(30, 31))
    
    Returns:
        data: JAX array of shape (..., N_meas)
        scheme: AcquisitionScheme object
    """
    if not DIPY_AVAILABLE:
        raise ImportError("DIPY is required to load Stanford HARDI data.")
        
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    
    data = img.get_fdata()
    bvals = gtab.bvals
    bvecs = gtab.bvecs
    
    # Create AcquisitionScheme
    # Note: DIPY bvals are usually in s/mm^2. dmipy-jax expects SI units (s/m^2) by default?
    # dmipy legacy uses SI. Let's convert if needed.
    # Standard convention in dmipy-jax is SI. 1 s/mm^2 = 1e6 s/m^2.
    bvals_si = bvals * 1e6
    
    # Create scheme
    # Dmipy legacy AcquisitionScheme requires delta/Delta/TE.
    # We use dummy values since standard datasets often lack them, and DTI/NODDI usually rely on b-value.
    # delta=0.01, Delta=0.02, TE=0.05 (seconds)
    scheme = acquisition_scheme_from_bvalues(bvals_si, bvecs, delta=0.01, Delta=0.02, TE=0.05)
    
    # Crop if requested
    if voxel_slice is not None:
        data = data[voxel_slice]
        
    return jnp.array(data), scheme

def load_sherbrooke_3shell(voxel_slice=None) -> Tuple[jnp.ndarray, Any]:
    """
    Loads the Sherbrooke 3-Shell dataset from DIPY.
    """
    if not DIPY_AVAILABLE:
         raise ImportError("DIPY is required to load Sherbrooke data.")

    fetch_sherbrooke_3shell()
    img, gtab = read_sherbrooke_3shell()
    
    data = img.get_fdata()
    bvals = gtab.bvals * 1e6 # Convert to SI
    bvecs = gtab.bvecs
    
    scheme = acquisition_scheme_from_bvalues(bvals, bvecs, delta=0.01, Delta=0.02, TE=0.05)
    
    if voxel_slice is not None:
        data = data[voxel_slice]
        
    return jnp.array(data), scheme

from dmipy_jax.data.bigmac import fetch_bigmac, get_bigmac_dwi_path
from dipy.io.image import load_nifti

def load_bigmac_mri(dataset_path: str = None, voxel_slice: Tuple[slice, ...] = None) -> Dict[str, Any]:
    """
    Loads the BigMac MRI data (DWI, Mask, T1, T2) into a dictionary.
    
    Args:
        dataset_path: Path to the BigMac dataset root. If None, fetches/uses cache.
        voxel_slice: Tuple of slices to crop the spatial dimensions.
        
    Returns:
        dict: {
            'dwi': np.ndarray (X, Y, Z, B),
            'scheme': JaxAcquisition,
            'mask': np.ndarray (X, Y, Z),
            'T1': np.ndarray (X, Y, Z),
            'T2': np.ndarray (X, Y, Z)
        }
    """
    check_dipy_installed()
    
    if dataset_path is None:
        dataset_path = fetch_bigmac()
    else:
        dataset_path = Path(dataset_path)
        
    if not dataset_path.exists():
         raise FileNotFoundError(f"BigMac dataset not found at {dataset_path}")

    results = {}
    
    # 1. DWI & Gradient Table
    try:
        dwi_path = get_bigmac_dwi_path(dataset_path)
        bval_path = str(dwi_path).replace(".nii.gz", ".bval")
        bvec_path = str(dwi_path).replace(".nii.gz", ".bvec")
        
        # Load NIfTI (eager load for now, slicing after)
        # For huge files, we should use nibabel slicing, but DIPY load_nifti is easier.
        data, affine = load_nifti(str(dwi_path))
        
        # Apply slice if requested
        if voxel_slice is not None:
            data = data[voxel_slice]
            
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        
        # Convert bvals to SI (s/m^2)
        if np.max(bvals) < 100000:
             bvals = bvals * 1e6
             
        results['dwi'] = jnp.array(data)
        results['scheme'] = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
        results['affine'] = affine
    except FileNotFoundError:
        print(f"Warning: DWI data not found in {dataset_path}")

    # 2. Structural Maps (T1/T2)
    # Search robustly in the dataset structure
    def find_and_load(pattern):
        files = list(dataset_path.rglob(pattern))
        if files:
            img, _ = load_nifti(str(files[0]))
            arr = img
            if voxel_slice is not None:
                arr = arr[voxel_slice]
            return jnp.array(arr)
        return None

    results['T1'] = find_and_load("*T1w.nii.gz")
    if results['T1'] is None:
        results['T1'] = find_and_load("*T1map.nii.gz")
        
    results['T2'] = find_and_load("*T2w.nii.gz")
    if results['T2'] is None:
        results['T2'] = find_and_load("*T2map.nii.gz")
        
    # 3. Mask
    results['mask'] = find_and_load("*mask.nii.gz")
    if results['mask'] is None and results['T1'] is not None:
        # Create rough mask from T1 > 0
        results['mask'] = results['T1'] > 0

    return results

def check_dipy_installed():
    if not DIPY_AVAILABLE:
        raise ImportError("DIPY is required for dataset loading. Install it via pip or uv.")

