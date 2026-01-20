
"""
Fetchers for public datasets requested for the 'CHAP' example.
Includes:
- Connectome 2.0 (ds006181)
- EDDEN (ds004666)
- Synaesthesia (ds004466)
"""
from pathlib import Path
from .openneuro import fetch_openneuro, DEFAULT_DATA_DIR

CONNECTOME2_ID = "ds006181"
EDDEN_ID = "ds004666"
SYNAESTHESIA_ID = "ds004466"

def fetch_connectome2(path: str = None) -> Path:
    """
    Fetches the Connectome 2.0 Diffusion MRI dataset (ds006181).
    High-gradient strength, advanced microstructure data.
    """
    return fetch_openneuro(CONNECTOME2_ID, path=path)

def fetch_edden(path: str = None) -> Path:
    """
    Fetches the EDDEN dataset (ds004666).
    Evaluation of Diffusion MRI DENoising.
    """
    return fetch_openneuro(EDDEN_ID, path=path)

def fetch_synaesthesia(path: str = None) -> Path:
    """
    Fetches the Synaesthesia Diffusion MRI dataset (ds004466).
    Multi-shell diffusion data.
    """
    return fetch_openneuro(SYNAESTHESIA_ID, path=path)

def get_dwi_path_generic(dataset_root: Path) -> Path:
    """
    Generic BIDS heuristic to find a DWI file.
    Returns the first *dwi.nii.gz found in su-*/dwi/.
    """
    # Strict BIDS
    candidates = list(dataset_root.glob("sub-*/ses-*/dwi/*dwi.nii.gz")) + \
                 list(dataset_root.glob("sub-*/dwi/*dwi.nii.gz"))
    
    if candidates:
        return candidates[0]
        
    # Relaxed
    candidates = list(dataset_root.rglob("*dwi.nii.gz"))
    if candidates:
        return candidates[0]
        
    raise FileNotFoundError(f"No DWI nifti files found in {dataset_root}")
