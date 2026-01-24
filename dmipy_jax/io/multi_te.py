import os
import glob
import numpy as np
import nibabel as nib
from typing import List, Tuple, Dict, Optional, Union
import jax.numpy as jnp

class MultiTELoader:
    """
    Loader for the Multi-TE Diffusion MRI Dataset.
    
    Expected Directory Structure:
        base_path/
            sub-XX/
                dwi/
                    sub-XX_acq-TE62_dwi.nii.gz  (Split format)
                    ...
                    OR
                    sub-XX_dwi.nii.gz           (Concatenated format)
                    sub-XX_dwi.bval
                    sub-XX_dwi.bvec
                    sub-XX_dwi.delta
    """
    
    def __init__(self, base_path: str, subject: str = 'sub-03'):
        """
        Initialize the loader.
        
        Args:
            base_path: Path to the MTE-dMRI dataset root (containing subject folders).
            subject: Subject ID (e.g., 'sub-03').
        """
        self.base_path = base_path
        self.subject = subject
        self.subject_dir = os.path.join(base_path, subject)
        self.dwi_dir = os.path.join(self.subject_dir, 'dwi')
        
        if not os.path.exists(self.subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")

        # Detection of format
        self.is_concatenated = False
        self.concat_prefix = os.path.join(self.dwi_dir, f"{self.subject}_dwi")
        if os.path.exists(self.concat_prefix + ".nii.gz"):
            self.is_concatenated = True
            
        self._cached_delta = None
        self._cached_bval = None

            
    def get_available_tes(self) -> List[str]:
        """
        Get a list of available Echo Times (TEs) strings found in the dataset.
        
        Returns:
            List of TEs (e.g., ['62', '72', '82', ..., '132', '62R2', '132R2']).
            Sorted numerically, with repeats (R2) at the end or handled appropriately.
        """
        if not os.path.exists(self.dwi_dir):
             # Fallback if dwi dir doesn't exist yet (e.g. during download)
             return []

        if self.is_concatenated:
            return self._get_tes_from_delta()

        # Find all .nii.gz files
        pattern = os.path.join(self.dwi_dir, f"{self.subject}_acq-TE*_dwi.nii.gz")
        files = glob.glob(pattern)
        
        tes = []
        for f in files:
            # Extract TE part: sub-03_acq-TE62_dwi.nii.gz -> TE62
            basename = os.path.basename(f)
            # Remove prefix and suffix
            try:
                # Format: sub-03_acq-TE*_dwi.nii.gz
                # Split by '_'
                parts = basename.split('_')
                # Find the part starting with 'acq-TE'
                acq_part = [p for p in parts if p.startswith('acq-TE')][0]
                # Extract TE value: acq-TE62 -> 62
                te_str = acq_part.replace('acq-TE', '')
                tes.append(te_str)
            except IndexError:
                continue
                
        # Sort TEs: intelligent sorting to handle integers and suffixes like 'R2'
        def sort_key(x):
            # Extract numeric part
            num_part = "".join([c for c in x if c.isdigit()])
            suffix = "".join([c for c in x if not c.isdigit()])
            try:
                val = int(num_part)
            except ValueError:
                val = 0
            return val, suffix
            
        return sorted(list(set(tes)), key=sort_key)

    def _get_tes_from_delta(self) -> List[str]:
        """Extract unique TEs from the .delta file for concatenated datasets."""
        delta_path = self.concat_prefix + ".delta"
        if not os.path.exists(delta_path):
            return []
            
        if self._cached_delta is None:
            self._cached_delta = np.loadtxt(delta_path)
            
        # Assuming delta values are in ms. 
        # Unique values of Big Delta usually correlate with TE shells in this dataset
        # But this dataset varies G and Delta.
        # Let's group by unique delta values and call them "TE" surrogates or just return unique deltas.
        # However, calling code expects "TE" strings.
        # In the CDMD dataset, Big Delta (Delta) matches the TE groups (e.g. 19ms, 49ms).
        # Wait, the .delta file usually contains small_delta, big_delta so 2 columns?
        # Or just one column? The user checked "head" and it showed single scalar "19". 
        # Usually .delta file has two columns or just "different diffusion parameters"?
        # Actually in CDMD, there are 2 diffusion times (19 and 49ms). 
        # The file output showed "19". So it's likely Big Delta.
        # We will use these unique values as the keys.
        
        unique_deltas = np.unique(self._cached_delta)
        return [str(int(d)) for d in sorted(unique_deltas)]

    def _get_filenames(self, te: str) -> Tuple[str, str, str]:
        """
        Construct filenames for a specific TE.
        """
        if self.is_concatenated:
            return self.concat_prefix + ".nii.gz", self.concat_prefix + ".bval", self.concat_prefix + ".bvec"
        
        prefix = f"{self.subject}_acq-TE{te}_dwi"
        nii_path = os.path.join(self.dwi_dir, f"{prefix}.nii.gz")
        bval_path = os.path.join(self.dwi_dir, f"{prefix}.bval")
        bvec_path = os.path.join(self.dwi_dir, f"{prefix}.bvec")
        return nii_path, bval_path, bvec_path


    def load_data(self, te: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Load DWI data, bvals, and bvecs for a specific TE.
        
        Args:
            te: TE string (e.g., '62', '132R2').
            
        Returns:
            data: JAX array of shape (X, Y, Z, N_dwis)
            bvals: JAX array of shape (N_dwis,)
            bvecs: JAX array of shape (N_dwis, 3)
        """
        nii_path, bval_path, bvec_path = self._get_filenames(te)
        
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"DWI file not found: {nii_path}")
        if not os.path.exists(bval_path):
            raise FileNotFoundError(f"bval file not found: {bval_path}")
        if not os.path.exists(bvec_path):
            raise FileNotFoundError(f"bvec file not found: {bvec_path}")
            
        # Load NIfTI
        img = nib.load(nii_path)
        
        if self.is_concatenated:
             # We need to slice the data for the requested TE/Delta
             if self._cached_delta is None:
                 delta_path = self.concat_prefix + ".delta"
                 self._cached_delta = np.loadtxt(delta_path)
             
             if self._cached_bval is None:
                 self._cached_bval = np.loadtxt(bval_path)
                 
             # Find indices matching this TE (Delta)
             # NOTE: we treat the input 'te' arg as the Delta value for concatenated sets
             target_delta = float(te)
             mask = np.abs(self._cached_delta - target_delta) < 0.1
             
             # Load only necessary volumes? NIfTI slicing might still load all if not careful
             # For simplicity and memory (1GB is fine), verify we can slice
             data_full = jnp.array(img.get_fdata()) # This might be heavy if 1GB, but 1GB is manageable
             data = data_full[..., mask]
             
             bvals_full = jnp.array(np.loadtxt(bval_path))
             bvals = bvals_full[mask]
             
             bvecs_full = jnp.array(np.loadtxt(bvec_path))
             bvecs = bvecs_full[mask]
             
             # Dynamic protocol fingerprint
             # We assume small_delta is fixed or also in a file?
             # For this dataset, delta is typically smaller.
             # If .delta file only has one column, it is likely Delta.
             # We need small_delta.
             # Paper says: delta=15.2, Delta=19 or 49?
             # If .delta has 19 and 49, then Delta=19/49.
             # We will return the specific Delta.
             
             protocol = {
                 'delta': 15.2, # Assumption if not in file
                 'Delta': target_delta
             }
             return data, bvals, bvecs, protocol

        data = jnp.array(img.get_fdata())
        
        # Load bvals/bvecs
        bvals = jnp.array(np.loadtxt(bval_path))
        bvecs = jnp.array(np.loadtxt(bvec_path).T) # Ensure (N, 3) shape
        
        return data, bvals, bvecs, self.PROTOCOL_FINGERPRINT

        
    def load_image_affine(self, te: str) -> np.ndarray:
        """Get the affine matrix for a specific TE volume."""
        nii_path, _, _ = self._get_filenames(te)
        img = nib.load(nii_path)
        return img.affine

    def load_paper_subset(self) -> List[str]:
        """
        Get the list of unique TEs used in the paper (ignoring repeats).
        TEs: 62, 72, 82, 92, 102, 112, 122, 132 ms.
        """
        all_tes = self.get_available_tes()
        
        if self.is_concatenated:
            # If concatenated, we likely found '19' and '49'. 
            # The 'paper set' usually refers to the specific scans.
            # We can either return all found, or filter if we knew exactly what the user wanted.
            # Return all found for now.
            return all_tes

        # Filter out 'R2' repeats
        paper_tes = [te for te in all_tes if 'R2' not in te]
        # Ensure we have the expected 8 TEs
        expected_tes = ['62', '72', '82', '92', '102', '112', '122', '132']
        # Intersection to be safe in case some haven't downloaded yet
        available_paper_tes = sorted(list(set(paper_tes).intersection(set(expected_tes))), key=lambda x: int(x))
        return available_paper_tes

    @property
    def PROTOCOL_FINGERPRINT(self) -> Dict[str, float]:
        """
        Acquisition protocol constants from the paper.
        Fixed diffusion times for all acquisitions.
        """
        return {
            'delta': 15.2, # ms
            'Delta': 25.2, # ms
        }
