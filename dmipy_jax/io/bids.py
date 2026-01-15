import os
import json
import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import nibabel as nib

# Attempt to import pybids
try:
    from bids.layout import BIDSLayout, parse_file_entities
    HAS_PYBIDS = True
except ImportError:
    HAS_PYBIDS = False
    BIDSLayout = None
    parse_file_entities = None

logger = logging.getLogger(__name__)

class BIDSLoader:
    def __init__(self, root_dir: str):
        if not HAS_PYBIDS:
            raise ImportError("pybids is required for BIDSLoader. Please install it with 'pip install pybids'.")
        
        self.root_dir = root_dir
        # Initialize layout. validate=False is often faster and sufficient for loading known data
        self.layout = BIDSLayout(root_dir, validate=False)

    def load_dwi(self, subject: str, session: Optional[str] = None) -> Dict[str, Any]:
        """
        Finds the DWI file, bval, and bvec for a given subject (and session).
        
        Args:
            subject: The subject ID (without 'sub-' prefix).
            session: Optional session ID (without 'ses-' prefix).
            
        Returns:
            Dict containing:
                - 'dwi_file': Path to the DWI NIfTI file.
                - 'bval_file': Path to the bval file.
                - 'bvec_file': Path to the bvec file.
                - 'EchoTime': Echo time (if found in metadata).
                - 'TotalReadoutTime': Total readout time (if found in metadata).
        """
        filters = {
            'subject': subject,
            'suffix': 'dwi',
            'extension': ['nii', 'nii.gz'],
            'return_type': 'file'
        }
        if session:
            filters['session'] = session
            
        dwi_files = self.layout.get(**filters)
        
        if not dwi_files:
            msg = f"No DWI files found for subject {subject}"
            if session:
                msg += f" session {session}"
            raise FileNotFoundError(msg)
            
        dwi_file = dwi_files[0]
        if len(dwi_files) > 1:
            logger.warning(f"Multiple DWI files found for sub-{subject}. Using {dwi_file}.")
            
        # Get bval/bvec
        # get_bval/bvec return absolute paths or None
        bval_file = self.layout.get_bval(dwi_file)
        bvec_file = self.layout.get_bvec(dwi_file)
        
        if not bval_file or not bvec_file:
            raise FileNotFoundError(f"Missing bval or bvec file for {dwi_file}")
            
        # Get metadata
        # get_metadata merges JSON sidecars from the hierarchy
        metadata = self.layout.get_metadata(dwi_file)
        
        result = {
            'dwi_file': dwi_file,
            'bval_file': bval_file,
            'bvec_file': bvec_file
        }
        
        # Extract crucial metadata
        if 'EchoTime' in metadata:
            result['EchoTime'] = metadata['EchoTime']
            
        if 'TotalReadoutTime' in metadata:
            result['TotalReadoutTime'] = metadata['TotalReadoutTime']
            
        return result


def save_bids_derivative(data: np.ndarray, 
                         input_file: str, 
                         metric_name: str, 
                         affine: Optional[np.ndarray] = None, 
                         header: Optional[Any] = None,
                         output_dir: Optional[str] = None) -> str:
    """
    Saves a result map (data) as a BIDS derivative.

    Args:
        data: The 3D or 4D array to save.
        input_file: The path to the source DWI file (used to extract entities).
        metric_name: The name of the metric/model (e.g., 'NODDI_ICVF').
                     Used in the filename as 'desc-{metric_name}'.
        affine: Optional affine matrix. If None, loaded from input_file.
        header: Optional header. If None, loaded from input_file.
        output_dir: Optional directory to save the file. 
                    If None, saves in the same directory as input_file.

    Returns:
        The absolute path to the saved NIfTI file.
    """
    if not HAS_PYBIDS:
        raise ImportError("pybids is required for save_bids_derivative.")
        
    # extract entities
    entities = parse_file_entities(input_file)
    
    # Construct filename parts
    # Requirement: extract entities (sub, ses, run).
    # Construct using desc-<ModelName> and correct suffix.
    
    parts = []
    if 'subject' in entities:
        parts.append(f"sub-{entities['subject']}")
    if 'session' in entities:
        parts.append(f"ses-{entities['session']}")
    if 'run' in entities:
        parts.append(f"run-{entities['run']}")
        
    # Add metric name as description
    parts.append(f"desc-{metric_name}")
    
    # Suffix - usually match input or 'dwi'
    suffix = entities.get('suffix', 'dwi')
    
    filename = "_".join(parts) + f"_{suffix}.nii.gz"
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, filename)
    else:
        out_path = os.path.join(os.path.dirname(input_file), filename)
        
    # Handle affine/header
    if affine is None or header is None:
        try:
            img = nib.load(input_file)
            if affine is None:
                affine = img.affine
            if header is None:
                header = img.header
        except Exception as e:
            logger.warning(f"Could not load input file {input_file} for affine/header: {e}")
            if affine is None:
                affine = np.eye(4)
                
    # Create and save image
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, out_path)
    
    # Save minimal JSON sidecar
    json_path = out_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    sidecar = {
        'Description': f'{metric_name} map derived from {os.path.basename(input_file)}',
        'Sources': [input_file],
        # Add basic provenance if needed
    }
    
    with open(json_path, 'w') as f:
        json.dump(sidecar, f, indent=4)
        
    return out_path
