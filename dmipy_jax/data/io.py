"""Low-level NIfTI and FSL gradient-table I/O utilities.

Provides thin wrappers around ``nibabel`` for reading NIfTI images and
FSL-format bval/bvec files into JAX arrays, and for writing parameter
maps back to NIfTI.

Key functions:

* :func:`load_nifti` -- read a ``.nii`` / ``.nii.gz`` into a
  ``jnp.ndarray``.
* :func:`load_bvals_bvecs` -- read FSL bval + bvec text files and return
  a :class:`~dmipy_jax.acquisition.JaxAcquisition` (b-values converted
  to SI s/m^2).
* :func:`save_nifti` -- write a NumPy/JAX array to NIfTI with a given
  affine.
"""

import jax.numpy as jnp
import nibabel as nib
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition
from pathlib import Path

def load_nifti(path: str) -> jnp.ndarray:
    """
    Loads a NIfTI file into a JAX array.
    
    Args:
        path: Path to .nii or .nii.gz file.
        
    Returns:
        jnp.ndarray containing the data.
    """
    img = nib.load(path)
    data = img.get_fdata()
    return jnp.array(data)

def load_bvals_bvecs(bval_path: str, bvec_path: str) -> JaxAcquisition:
    """
    Loads b-values and b-vectors from FSL-style text files and creates a JaxAcquisition.
    
    Args:
        bval_path: Path to .bval file.
        bvec_path: Path to .bvec file.
        
    Returns:
        JaxAcquisition object.
    """
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T # Transpose to shape (N, 3) usually?
    # FSL bvecs are usually (3, N), dmipy expects (N, 3).
    # Double check shape.
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
        
    # Check dimensions
    if len(bvals) != bvecs.shape[0]:
         # Attempt transpose if mismatch
         if len(bvals) == bvecs.shape[1]:
              bvecs = bvecs.T
         else:
             raise ValueError(f"Shape mismatch: bvals {len(bvals)}, bvecs {bvecs.shape}")

    # Create JaxAcquisition
    # Note: dmipy_jax models generally expect SI units?
    # FSL bvals are usually s/mm^2. dmipy/dmipy_jax usually wants SI (s/m^2).
    # Check convention. dmipy usually converts e.g. 1000 -> 1000e6.
    
    # Heuristic check for units
    if np.max(bvals) < 10000:
        # Likely s/mm^2, convert to s/m^2
        bvals = bvals * 1e6

    return JaxAcquisition(bvalues=jnp.array(bvals), gradient_directions=jnp.array(bvecs))


def save_nifti(data, affine, path: str) -> None:
    """Save an array as a NIfTI file.

    Parameters
    ----------
    data : array-like
        Image data (3-D or 4-D).
    affine : np.ndarray
        4x4 affine matrix.
    path : str
        Output file path (``.nii`` or ``.nii.gz``).
    """
    data_np = np.asarray(data, dtype=np.float32)
    img = nib.Nifti1Image(data_np, affine)
    nib.save(img, path)
