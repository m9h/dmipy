
import os
import glob
import numpy as np
import nibabel as nib
import jax.numpy as jnp
from typing import Tuple, List, Dict

class MicaMICsLoader:
    """
    Loader for the MICA-MICs Dataset.
    
    Expected Structure:
        base_path/
            MICs_release/
                rawdata/
                    sub-HC001/
                        ses-01/
                            dwi/
                                sub-HC001_ses-01_acq-b300-11_dir-AP_dwi.nii.gz
                                ...
    """
    
    def __init__(self, base_path: str, subject: str = 'sub-HC001', session: str = 'ses-01'):
        self.base_path = base_path
        self.subject = subject
        self.session = session
        
        # Path constructor
        self.dwi_dir = os.path.join(
            base_path, 
            'MICs_release', 'rawdata', subject, session, 'dwi'
        )
        
        if not os.path.exists(self.dwi_dir):
            raise FileNotFoundError(f"DWI directory not found: {self.dwi_dir}")

    def load_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Loads all available DWI shells and concatenates them.
        
        Returns:
            data: (X, Y, Z, N) JAX array
            bvals: (N,) JAX array (s/mm^2 usually, verification needed)
            bvecs: (N, 3) JAX array
        """
        # Find all .nii.gz files in dwi folder
        pattern = os.path.join(self.dwi_dir, "*.nii.gz")
        files =  sorted(glob.glob(pattern))
        
        all_data = []
        all_bvals = []
        all_bvecs = []
        
        print(f"Found {len(files)} DWI files.")
        
        for nii_path in files:
            # Assume paired .bval and .bvec exist
            basename = nii_path.replace('.nii.gz', '')
            bval_path = basename + '.bval'
            bvec_path = basename + '.bvec'
            
            if not os.path.exists(bval_path) or not os.path.exists(bvec_path):
                print(f"Skipping {nii_path}: Missing bval/bvec")
                continue
                
            print(f"Loading {os.path.basename(nii_path)}...")
            
            # Load NIfTI
            img = nib.load(nii_path)
            data_chunk = jnp.array(img.get_fdata())
            
            # Load Bvals/Bvecs
            # Note: bvals are typically 1D, bvecs 3xN or Nx3
            bvals_chunk = np.loadtxt(bval_path)
            bvecs_chunk = np.loadtxt(bvec_path)
            
            # Check dimensions make sense
            # Image is (X,Y,Z, T)
            if data_chunk.ndim == 4:
                n_vols = data_chunk.shape[-1]
            else:
                n_vols = 1
                data_chunk = data_chunk[..., None]
                
            # Verify bvals shape
            if bvals_chunk.size != n_vols:
                print(f"Warning: bvals count {bvals_chunk.size} != volumes {n_vols} in {os.path.basename(nii_path)}")
                
            # Verify bvecs shape and transpose if necessary
            # We expect (N, 3) for dmipy-jax
            if bvecs_chunk.shape[0] == 3 and bvecs_chunk.shape[1] == n_vols:
                bvecs_chunk = bvecs_chunk.T # Convert 3xN to Nx3
            elif bvecs_chunk.shape[0] == n_vols and bvecs_chunk.shape[1] == 3:
                pass # Already Nx3
            else:
                 print(f"Warning: bvecs shape {bvecs_chunk.shape} unclear for {n_vols} volumes.")
                 
            all_data.append(data_chunk)
            all_bvals.append(jnp.array(bvals_chunk))
            all_bvecs.append(jnp.array(bvecs_chunk))
            
        # Concatenate
        full_data = jnp.concatenate(all_data, axis=-1)
        full_bvals = jnp.concatenate(all_bvals, axis=0)
        full_bvecs = jnp.concatenate(all_bvecs, axis=0)
        
        return full_data, full_bvals, full_bvecs
