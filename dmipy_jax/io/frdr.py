"""FRDR C57Bl/6 longitudinal mouse brain MRI dataset loader.

Provides :class:`FRDRMouseLoader` for accessing the CAMRI/CFMM
longitudinal mouse dataset from the Federated Research Data Repository
(FRDR).  The dataset contains multi-session (longitudinal) structural
and diffusion MRI of C57Bl/6 mice in BIDS-like format.

Useful for preclinical microstructure validation studies where
ground-truth histology or known anatomy can constrain model evaluation.
"""

import os
import glob
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FRDRMouseLoader:
    """
    Loader for the FRDR C57Bl/6 Longitudinal Mouse Dataset (CAMRI/CFMM).
    URL: https://www.frdr-dfdr.ca/repo/dataset/9ea832ad-7f36-4e37-b7ac-47167c0001c1
    
    This dataset typically follows a structure like:
    root/
      sub-01/
        ses-01/
          anat/
          dwi/
    """
    
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        if not os.path.exists(self.root_dir):
            logger.warning(f"FRDR Root directory {self.root_dir} does not exist.")

    def get_subjects(self):
        """Returns a list of subject IDs found in the root directory."""
        # Assume BIDS-like 'sub-XX' folders
        sub_dirs = glob.glob(os.path.join(self.root_dir, "sub-*"))
        return [os.path.basename(d).replace("sub-", "") for d in sub_dirs]

    def load_subject(self, subject: str, session: str) -> Dict[str, Any]:
        """
        Locates files for a specific subject and session.
        
        Args:
            subject: Subject ID string (e.g., '01')
            session: Session ID string (e.g., '01')
            
        Returns:
            Dict containing paths to:
                - 'dwi_nii': DWI NIfTI path
                - 'bval': bval path
                - 'bvec': bvec path
                - 'anat_nii': T2w/Anatomical NIfTI path
                - 'mask_nii': Brain mask NIfTI path (if available)
        """
        base_path = os.path.join(self.root_dir, f"sub-{subject}", f"ses-{session}")
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Subject directory not found: {base_path}")
            
        # Define search patterns for this specific dataset
        # Note: Adjusting extensions to allow .nii or .nii.gz
        
        # DWI
        # Looking for something like sub-01_ses-01_dwi.nii.gz
        dwi_glob = glob.glob(os.path.join(base_path, "dwi", f"*dwi.nii*"))
        if not dwi_glob:
             # Fallback: maybe flat structure?
             dwi_glob = glob.glob(os.path.join(base_path, f"*dwi.nii*"))
        
        if not dwi_glob:
            raise FileNotFoundError(f"No DWI nifti found for sub-{subject} ses-{session}")
        dwi_nii = dwi_glob[0]
        
        # Bvals/Bvecs
        # Often named similarly to the DWI file
        base_name = dwi_nii.split(".nii")[0]
        bval = base_name + ".bval"
        bvec = base_name + ".bvec"
        
        if not os.path.exists(bval):
             # Try simpler pattern
             bval_glob = glob.glob(os.path.join(os.path.dirname(dwi_nii), "*.bval"))
             if bval_glob: bval = bval_glob[0]
        
        if not os.path.exists(bvec):
             # Try simpler pattern
             bvec_glob = glob.glob(os.path.join(os.path.dirname(dwi_nii), "*.bvec"))
             if bvec_glob: bvec = bvec_glob[0]
             
        # Anatomical (T2w)
        # Looking for sub-01_ses-01_T2w.nii.gz
        anat_glob = glob.glob(os.path.join(base_path, "anat", f"*T2w.nii*"))
        if not anat_glob:
            anat_glob = glob.glob(os.path.join(base_path, f"*T2w.nii*"))
            
        anat_nii = anat_glob[0] if anat_glob else None
        
        return {
            "dwi_nii": dwi_nii,
            "bval": bval,
            "bvec": bvec,
            "anat_nii": anat_nii
        }
