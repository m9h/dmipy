"""Allen Mouse Brain Common Coordinate Framework (CCFv3) loader.

Provides :class:`AllenCCFLoader` for downloading and loading the Allen
Institute's CCFv3 atlas volumes, including:

* Average template (50 um or 25 um isotropic).
* Annotation volume (voxelwise region labels).

These volumes are used for registering diffusion data to the Allen atlas
coordinate space, enabling region-of-interest analysis and connectome
mapping in preclinical (rodent) diffusion MRI studies.
"""

import os
import logging
import urllib.request
import gzip
import shutil
import nibabel as nib
from pathlib import Path

logger = logging.getLogger(__name__)

class AllenCCFLoader:
    """
    Loader for the Allen Mouse Brain Common Coordinate Framework (CCFv3).
    
    Provides access to:
    - Average Template (50um, 25um)
    - Annotation Volume (Segmentation Labels)
    """
    
    # Public URLs for CCFv3 NIfTIs (using a reliable mirror or Allen direct if available)
    # Using BrainGlobe's cached URLs or similar is a good standard approach
    # For this implementation, we will expect them to be provided or guide the user
    # But to make it "easy", we can try to fetch a 50um version if typically available.
    
    # Example valid URLs (often from NITRC or similar repositories for NIfTI converted versions)
    # Since direct hotlinking can be unstable, we prioritize a local cache.
    
    def __init__(self, cache_dir: str = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~/.dmipy_jax/data/allen_ccf"))
            
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_ccfv3(self, resolution_um: int = 50) -> dict:
        """
        Ensures CCFv3 template and annotation are available locally.
        
        Args:
            resolution_um: Resolution in microns (50 or 25).
            
        Returns:
            dict: {'template': Path, 'annotation': Path}
        """
        if resolution_um not in [25, 50]:
            raise ValueError("Supported resolutions are 25 or 50 microns.")
            
        template_name = f"allen_mouse_ccf_v3_{resolution_um}um_template.nii.gz"
        annotation_name = f"allen_mouse_ccf_v3_{resolution_um}um_annotation.nii.gz"
        
        template_path = self.cache_dir / template_name
        annotation_path = self.cache_dir / annotation_name
        
        if not template_path.exists() or not annotation_path.exists():
            logger.info(f"Allen CCFv3 ({resolution_um}um) not found in {self.cache_dir}")
            logger.info("Attempting automatic download...")
            
            try:
                self._download_ccf(resolution_um, template_path, annotation_path)
            except Exception as e:
                logger.error(f"Failed to download CCFv3: {e}")
                logger.error(f"Please manually download the {resolution_um}um NIfTI files to {self.cache_dir}")
                raise
                
        return {
            "template": str(template_path),
            "annotation": str(annotation_path)
        }

    def _download_ccf(self, resolution, t_path, a_path):
        """
        Downloads CCFv3 data from official Allen Institute archive.
        """
        # Official URLs for 50um and 25um
        # Note: 25um is much larger
        
        base_url = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf"
        
        urls = {
            50: {
                "template": f"{base_url}/average_template/average_template_50.nii.gz",
                "annotation": f"{base_url}/annotation/ccf_2017/annotation_50.nii.gz"
            },
            25: {
                "template": f"{base_url}/average_template/average_template_25.nii.gz",
                "annotation": f"{base_url}/annotation/ccf_2017/annotation_25.nii.gz"
            }
        }
        
        if resolution not in urls:
            raise ValueError(f"No direct URL for {resolution}um. Only 25 and 50 are supported.")
            
        logger.info(f"Downloading CCFv3 {resolution}um Template...")
        self._download_file(urls[resolution]["template"], t_path)
        
        logger.info(f"Downloading CCFv3 {resolution}um Annotation...")
        self._download_file(urls[resolution]["annotation"], a_path)
        
    def _download_file(self, url, dest_path):
        try:
            with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info(f"Downloaded to {dest_path}")
        except Exception as e:
            # Clean up partial
            if dest_path.exists():
                dest_path.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}")

    def load_labels(self):
        """
        Loads the mapping of ID -> Region Name (Ontology).
        """
        # This would typically load the ontology CSV/JSON.
        # For now, we'll return a stub or implement a parser if we download the CSV.
        return {}
