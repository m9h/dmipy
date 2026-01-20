import os
import logging
from pathlib import Path

# Try to import datalad
try:
    import datalad.api as dl
    HAS_DATALAD = True
except ImportError:
    HAS_DATALAD = False

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".cache" / "dmipy_jax" / "data"

def fetch_datalad(source_url: str, path: Path) -> Path:
    """
    Fetches a dataset using DataLad from a generic URL.
    
    Args:
        source_url: Git/DataLad URL of the source dataset.
        path: Local destination path.
        
    Returns:
        Path to the installed dataset.
    """
    if not HAS_DATALAD:
        raise ImportError(
            "DataLad is required to fetch datasets. "
            "Please install it via pip or apt (datalad) and ensure git-annex is available."
        )
        
    path = Path(path)
    
    if path.exists():
        logger.info(f"Dataset found at {path}. Updating...")
        try:
            ds = dl.Dataset(str(path))
            if ds.is_installed():
                ds.update(merge=True)
                return path
            else:
                 logger.warning(f"Path {path} exists but is not a valid DataLad dataset. Proceeding to install.")
        except Exception as e:
            logger.warning(f"Error checking existing dataset at {path}: {e}")

    logger.info(f"Installing from {source_url} to {path}")
    dl.install(source=source_url, path=str(path))
    return path

def fetch_openneuro(dataset_id: str, path: str = None, version: str = None) -> Path:
    """
    Fetches a dataset from OpenNeuro using DataLad.
    
    Args:
        dataset_id: OpenNeuro Accession Number (e.g., 'ds005089').
        path: Destination path. Defaults to ~/.cache/dmipy_jax/data/<dataset_id>.
        version: Specific version tag (optional).
        
    Returns:
        Path to the installed dataset.
    """
    if path is None:
        path = DEFAULT_DATA_DIR / dataset_id
    else:
        path = Path(path)
        
    source_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    return fetch_datalad(source_url, path)
