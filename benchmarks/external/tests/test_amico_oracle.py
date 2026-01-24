import numpy as np
import pytest
import os
import sys

# Ensure benchmarks/external is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from external.amico_oracle import AmicoOracle

def test_amico_oracle_noddi_build_and_run():
    """
    Test that the AMICO Oracle can be initialized (builds Docker)
    and run on a small synthetic dataset.
    """
    # 1. Initialize Oracle
    print("Initializing Oracle (this may take time to build Docker image)...")
    oracle = AmicoOracle()
    
    # 2. Generate Synthetic Data
    # Shape: (2, 2, 2, 30) - very small for speed
    # 30 gradient directions (approx)
    n_b0 = 3
    n_diff = 27
    n_total = n_b0 + n_diff
    
    data = np.random.rand(2, 2, 2, n_total) * 1000 # Random signal
    # Avoid zeros to prevent div by zero in some models
    data += 100 
    
    # b-values: 0s and 1000s
    bvals = np.concatenate([np.zeros(n_b0), np.ones(n_diff) * 1000])
    
    # b-vecs: random unit vectors
    bvecs = np.random.randn(n_total, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    
    # 3. fit
    print("Running Fit...")
    # Using NODDI as default
    results = oracle.fit(data, bvals, bvecs, model="NODDI")
    
    # 4. Verify
    print("Results keys:", results.keys())
    assert len(results) > 0
    # Check for expected NODDI outputs
    assert "ICVF" in results or "ficvf" in results # capitalization might vary
    assert "OD" in results or "odi" in results
    
    print("Test passed!")

if __name__ == "__main__":
    test_amico_oracle_noddi_build_and_run()
