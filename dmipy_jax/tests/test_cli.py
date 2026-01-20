import pytest
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from dmipy_jax.cli import bids_report

@pytest.fixture
def mock_bids_dataset(tmp_path):
    root = tmp_path / "bids_root"
    root.mkdir()
    
    # Description
    with open(root / "dataset_description.json", "w") as f:
        json.dump({"Name": "Test Dataset", "BIDSVersion": "1.0.0"}, f)
        
    # Subject 1
    sub1 = root / "sub-01"
    (sub1 / "dwi").mkdir(parents=True)
    (sub1 / "anat").mkdir(parents=True)
    
    # Create Dummy DWI Nifti
    data = np.zeros((10, 10, 10, 10))
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, sub1 / "dwi" / "sub-01_dwi.nii.gz")
    
    # Create Dummy bvals (2 shells: b=0 (1), b=1000 (3), b=2000 (6)) -> Total 10
    bvals = np.array([0, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000])
    np.savetxt(sub1 / "dwi" / "sub-01_dwi.bval", bvals, fmt="%d") # 1D
    
    # Subject 2 (Anat only)
    sub2 = root / "sub-02"
    (sub2 / "anat").mkdir(parents=True)
    nib.save(img, sub2 / "anat" / "sub-02_T1w.nii.gz")
    
    return root

def test_find_bids_root(mock_bids_dataset):
    # Test from root
    assert bids_report.find_bids_root(mock_bids_dataset) == mock_bids_dataset
    
    # Test from sub-folder
    assert bids_report.find_bids_root(mock_bids_dataset / "sub-01") == mock_bids_dataset

def test_analyze_shells(tmp_path):
    bval_path = tmp_path / "test.bval"
    
    # Test Normal
    np.savetxt(bval_path, [0, 995, 1005, 2000]) # 995 and 1005 should cluster to 1000
    shells = bids_report.analyze_shells(bval_path)
    assert shells == {0: 1, 1000: 2, 2000: 1}
    
    # Test Empty/Error
    assert bids_report.analyze_shells(tmp_path / "nonexistent") == {}

def test_generate_report(mock_bids_dataset):
    report = bids_report.generate_report(mock_bids_dataset)
    
    print(report) # For debugging in failure case
    
    assert "Test Dataset" in report
    assert "Total Subjects: 2" in report
    assert "dwi" in report
    assert "anat" in report
    assert "b=1000  : 3" in report or "b=1000 : 3" in report # Whitespace tolerant check
    assert "b=2000  : 6" in report or "b=2000 : 6" in report

def test_main(mock_bids_dataset, capsys):
    # Mock sys.argv
    import sys
    from unittest.mock import patch
    
    args = ["dmipy-report", str(mock_bids_dataset)]
    with patch.object(sys, 'argv', args):
        bids_report.main()
    
    captured = capsys.readouterr()
    assert "BIDS Dataset Report" in captured.out
    assert "Total Subjects: 2" in captured.out
