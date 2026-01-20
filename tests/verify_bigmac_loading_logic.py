
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
import numpy as np

# Mock dependencies to test logic without downloading 10GB
from dmipy_jax.io.datasets import load_bigmac_mri

def test_load_bigmac_logic_mocks():
    """
    Verifies that load_bigmac_mri correctly orchestrates fetching and loading
    using mocked file system and openneuro calls.
    """
    with patch("dmipy_jax.io.datasets.fetch_bigmac") as mock_fetch, \
         patch("dmipy_jax.io.datasets.get_bigmac_dwi_path") as mock_get_path, \
         patch("dmipy_jax.io.datasets.load_nifti") as mock_load_nifti, \
         patch("dmipy_jax.io.datasets.read_bvals_bvecs") as mock_read_bv, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.rglob") as mock_rglob:
         
        # Setup Mocks
        mock_root = Path("/mock/bigmac")
        mock_fetch.return_value = mock_root
        mock_exists.return_value = True
        
        # Mock DWI path
        mock_dwi = mock_root / "dwi" / "sub-BigMac_dwi.nii.gz"
        mock_get_path.return_value = mock_dwi
        
        # Mock Data Return
        mock_load_nifti.return_value = (np.zeros((10, 10, 10, 30)), np.eye(4))
        mock_read_bv.return_value = (np.ones(30)*1000, np.zeros((30, 3)))
        
        # Mock T1 search
        # first call to rglob (*T1w) returns list
        mock_rglob.side_effect = [
            [mock_root / "anat" / "T1w.nii.gz"], # T1
            [mock_root / "anat" / "T2w.nii.gz"], # T2
            [] # Mask (not found, triggers fallback)
        ]
        
        # Execute
        data = load_bigmac_mri()
        
        # Assertions
        mock_fetch.assert_called_once()
        mock_get_path.assert_called_with(mock_root)
        assert 'dwi' in data
        assert 'T1' in data
        assert 'T2' in data
        assert 'scheme' in data
        # Check mask fallback logic (from T1)
        assert 'mask' in data
        assert data['mask'].shape == (10, 10, 10, 30) # Wait, T1 is 4D? No usually 3D. 
        # But mock returned 4D for all load_nifti.
        # It's fine for logic test.

if __name__ == "__main__":
    test_load_bigmac_logic_mocks()
    print("Logic verification passed.")
