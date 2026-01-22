import pytest
import numpy as np
import nibabel as nib
import jax.numpy as jnp
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

from dmipy_jax.data import io, openneuro, bigmac

def test_load_nifti(tmp_path):
    # Create dummy NIfTI
    data = np.random.rand(10, 10, 10)
    img = nib.Nifti1Image(data, np.eye(4))
    fpath = tmp_path / "test.nii.gz"
    nib.save(img, fpath)
    
    # Load
    loaded = io.load_nifti(str(fpath))
    assert isinstance(loaded, jnp.ndarray)
    assert np.allclose(loaded, data)

def test_load_bvals_bvecs(tmp_path):
    # Create dummy bvals/bvecs
    bvals = np.array([0, 1000, 2000])
    bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).T # (3, N)
    
    bval_path = tmp_path / "test.bval"
    bvec_path = tmp_path / "test.bvec"
    
    np.savetxt(bval_path, bvals)
    np.savetxt(bvec_path, bvecs)
    
    # Load
    acq = io.load_bvals_bvecs(str(bval_path), str(bvec_path))
    
    # Check scaling (heuristic < 10000 -> *1e6)
    assert jnp.allclose(acq.bvalues, bvals * 1e6)
    assert acq.gradient_directions.shape == (3, 3)

def test_fetch_openneuro_mock():
    with patch("dmipy_jax.data.openneuro.dl") as mock_dl:
        mock_dl.install = MagicMock()
        
        # Test fetch
        openneuro.fetch_openneuro("dsTest", path="/tmp/test")
        
        mock_dl.install.assert_called_once()
        args = mock_dl.install.call_args[1]
        assert args['source'] == "https://github.com/OpenNeuroDatasets/dsTest.git"
        assert args['path'] == "/tmp/test"

def test_fetch_bigmac_integration():
    with patch("dmipy_jax.data.bigmac.fetch_datalad") as mock_fetch:
        bigmac.fetch_bigmac("/tmp/bigmac")
        mock_fetch.assert_called_once_with(bigmac.BIGMAC_URL, path="/tmp/bigmac")
