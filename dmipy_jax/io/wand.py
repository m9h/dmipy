
import os
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from dmipy_jax.data.openneuro import fetch_datalad, DEFAULT_DATA_DIR

# WAND GIN Repository
WAND_GIN_URL = "https://gin.g-node.org/CUBRIC/WAND"

class WANDLoader:
    def __init__(self, base_path=None, subject="sub-01"):
        if base_path is None:
            self.base_path = DEFAULT_DATA_DIR / "WAND"
        else:
            self.base_path = Path(base_path)
            
        self.subject = subject
        self.bids_root = self.base_path
        
    def fetch_data(self):
        """
        Fetches WAND dataset from GIN if not present.
        """
        if not self.base_path.exists():
            print(f"Fetching WAND dataset to {self.base_path}...")
            fetch_datalad(WAND_GIN_URL, path=self.base_path)
            
    def get_axcaliber_files(self, session="ses-01"):
        """
        Finds AxCaliber acquisitions for the subject.
        Looking for 'acq-AxCaliber1' and 'acq-AxCaliber2'.
        """
        subj_dir = self.bids_root / self.subject / session / "dwi"
        if not subj_dir.exists():
            # Try searching without session if flat?
            # Creating list of possible locations
            candidates = [
                self.bids_root / self.subject / "dwi",
                self.bids_root / "rawdata" / self.subject / session / "dwi"
            ]
            for p in candidates:
                if p.exists():
                    subj_dir = p
                    break
            else:
                 raise FileNotFoundError(f"Could not find DWI directory for {self.subject}")
        
        # Files
        # Pattern: sub-01_ses-01_acq-AxCaliber1_dwi.nii.gz
        # We look for glob *acq-AxCaliber*_dwi.nii.gz
        
        files = sorted(list(subj_dir.glob(f"*{self.subject}*acq-AxCaliber*dwi.nii.gz")))
        # Filter out Ref files
        files = [f for f in files if "Ref" not in f.name]
        print(f"Found {len(files)} AxCaliber files (excluding Ref): {[f.name for f in files]}")
        return files

    def load_axcaliber_data(self, session="ses-01", roi_slice=None):
        """
        Loads and concatenates AxCaliber data.
        Returns:
            data: (X, Y, Z, N)
            bvals: (N,)
            bvecs: (N, 3)
            big_delta: (N,)
            small_delta: (N,)
        """
        files = self.get_axcaliber_files(session)
        if not files:
            raise FileNotFoundError("No AxCaliber files found.")
            
        all_data = []
        all_bvals = []
        all_bvecs = []
        all_big = []
        all_small = []
        
        for fpath in files:
            print(f"Loading {fpath.name}...")
            # Load NIfTI
            data, affine = load_nifti(str(fpath))
            if roi_slice:
                data = data[roi_slice]
                
            # Load bvals/bvecs
            bval_path = str(fpath).replace(".nii.gz", ".bval")
            bvec_path = str(fpath).replace(".nii.gz", ".bvec")
            bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
            
            # Load JSON for Delta/delta
            json_path = str(fpath).replace(".nii.gz", ".json")
            if not os.path.exists(json_path):
                 print(f"Warning: JSON not found for {fpath.name}. Using defaults.")
                 delta = 0.0 # Placeholder
                 Delta = 0.0
            else:
                with open(json_path, 'r') as jf:
                    meta = json.load(jf)
                    
                # Parse WAND specific keys
                # "t_bdel": Big Delta (ms?)
                # "t_sdel": Small Delta (ms?)
                # Or standard BIDS fields?
                
                # Check standard fields first
                # Usually in BIDS these might not be standard. WAND uses custom keys?
                # Based on description: t_bdel
                
                # Let's look for keys
                if 't_bdel' in meta:
                    Delta = float(meta['t_bdel']) # Assuming seconds or need conversion?
                    # Usually datasets store SI or ms. Let's assume ms if > 1?
                    # WAND 300mT/m scanner often reports in s?
                    # Let's check magnitude. If > 1, likely ms.
                    if Delta > 1.0: Delta *= 1e-3
                elif 'DiffusionGradientPulseDuration' in meta:
                     # Standard BIDS?
                     pass
                else:
                    Delta = 0.03 # Default guess 30ms

                if 't_sdel' in meta:
                    delta = float(meta['t_sdel'])
                    if delta > 1.0: delta *= 1e-3
                else:
                    delta = 0.01 # Default guess 10ms
            
            n_vol = len(bvals)
            all_data.append(data)
            all_bvals.append(bvals)
            all_bvecs.append(bvecs)
            all_big.append(jnp.full((n_vol,), Delta))
            all_small.append(jnp.full((n_vol,), delta))
            
        # Concatenate
        full_data = jnp.concatenate(all_data, axis=-1)
        full_bvals = jnp.concatenate(all_bvals, axis=0) * 1e6 # s/mm2 -> s/m2 if needed?
        # Usually bvals are s/mm2 (~1000). SI requires s/m2 (~1e9).
        # We apply conversion here if raw is standard.
        
        full_bvecs = jnp.concatenate(all_bvecs, axis=0)
        full_big = jnp.concatenate(all_big, axis=0)
        full_small = jnp.concatenate(all_small, axis=0)
        
        return {
            'data': full_data,
            'bvals': full_bvals,
            'bvecs': full_bvecs, 
            'big_delta': full_big,
            'small_delta': full_small,
            'affine': affine
        }
