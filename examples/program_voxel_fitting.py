
"""
Testing Program Part 1: Voxel Fitting (Optimistix Showcase).

Duplicates the Connectome 2.0 pipeline on 'Synesthesia' synthetic data.
Focus: Recovering Axon Diameter Gradient and Crossing Fibers.
"""

import jax
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# ensure benchmarks module is importable
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../benchmarks'))
from generate_synthetic_hcp import generate_synthetic_connectome

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.fitting.optimization import OptimistixFitter

def run_voxel_program():
    print("=== Testing Program: Optimistix Voxel Fitting ===")
    
    # 1. Generate 'Synesthesia' Data
    data, scheme, gt_params = generate_synthetic_connectome(shape=(20, 20, 1), snr=50) # Low Z for speed
    
    # 2. Define Fitting Model
    # We fit a simpler model than ground truth to test robustness?
    # Or strict duplicate of pipeline? C2 pipeline used RestrictedCylinder+Zeppelin+Ball.
    # Our GT is RestrictedCylinder+RestrictedCylinder+Ball.
    # Let's fit Single Fiber model first to see if we recover diameter in Zone 1.
    
    print("\n[Test 1] Fitting Single Axon Diameter Model (Zone 1 Focus)")
    intra = cylinder_models.RestrictedCylinder()
    csf = gaussian_models.Ball()
    model_s = JaxMultiCompartmentModel([intra, csf])
    
    print("Fitting slice...")
    fitted = model_s.fit(scheme, data, method="Optimistix")
    
    # 3. Validation
    # Check Diameter recovery in Zone 1
    # Model: RestrictedCylinder (0) + Ball (1). Key is 'diameter'.
    diam_est_flat = fitted['diameter'] # (N_vox,)
    # If using OptimistixFitter directly or fit(), checks shape.
    if diam_est_flat.ndim > 1 and diam_est_flat.shape[-1] == 1:
        diam_est_flat = diam_est_flat[..., 0]
        
    diam_est = diam_est_flat.reshape(data.shape[:-1]) # (20, 20, 1)
    
    diam_gt = gt_params['diameter'][..., 0]
    
    mask_z1 = (diam_gt > 2.1e-6) # Zone 1 roughly
    
    err = np.abs(diam_est[mask_z1] - diam_gt[mask_z1])
    mae = np.mean(err)
    print(f"Zone 1 Diameter MAE: {mae*1e6:.2f} um")
    
    if mae < 1.0e-6:
        print("SUCCESS: Diameter gradient recovered within 1um tolerance.")
    else:
        print("WARNING: Diameter recovery poor.")
        
    print("\n[Test Complete]")

if __name__ == "__main__":
    run_voxel_program()
