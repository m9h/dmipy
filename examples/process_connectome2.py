
"""
Advanced Processing Pipeline for Connectome 2.0 (ds006181).
Leverages 'dmipy-jax' high-performance tools:
- RestrictedCylinder model (Soderman/Callaghan) for High-G sensitivity.
- OptimistixFitter for robust non-linear optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dmipy_jax.io.connectome2 import load_connectome2_dwi
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.fitting.optimistix_fitter import OptimistixFitter

def main():
    print("--- Connectome 2.0 Advanced Processing ---")
    
    # 1. Load Data (Center Slice for Demo)
    # We crop to a small central ROI to ensure the demo runs fast.
    print("Loading data (Center Slice)...")
    # Note: We don't know the exact dimensions until we load, 
    # so we load a bit generically then crop, or rely on the loader's laziness if any.
    # The loader is eager for now, but we pass voxel_slice to crop post-load or during load.
    
    # Let's load the whole thing first to inspect shape (it's cached anyway)
    # Or better, just load a slice blindly assuming standard brain size
    data_dict = load_connectome2_dwi()
    data = data_dict['dwi']
    scheme = data_dict['scheme']
    
    print(f"Data Shape: {data.shape}")
    print(f"B-values: {np.unique(scheme.bvalues.round(-2))}")
    
    mid_x, mid_y, mid_z = np.array(data.shape[:3]) // 2
    roi_data = data[mid_x-10:mid_x+10, mid_y-10:mid_y+10, mid_z:mid_z+1, :]
    
    # 2. Define Advanced Model
    # Connectome 2.0 is ideal for measuring axon diameter.
    # We use RestrictedCylinder (Cylinder with finite radius)
    # Note: This requires Delta/delta to be present in the scheme.
    if scheme.delta is None or scheme.Delta is None:
        print("[!] Warning: Delta/delta not found in metadata. Defaulting to estimated values for demo.")
        scheme.delta = 0.010 # 10ms
        scheme.Delta = 0.030 # 30ms
    
    print("Defining Model: RestrictedCylinder + Zeppelin + Ball")
    # Restricted Cylinder: Intra-axonal
    # Connectome 2.0 High-G data allows estimation of axon diameter.
    intra = cylinder_models.RestrictedCylinder(diameter=None) # Allow fitting diameter
    extra = gaussian_models.C2Zeppelin()
    csf = gaussian_models.C3Ball()
    
    model = JaxMultiCompartmentModel(models=[intra, extra, csf])
    
    # 3. Fit
    print("Fitting with OptimistixFitter (Levenberg-Marquardt)...")
    fitter = OptimistixFitter(model, scheme)
    fitted_params = fitter.fit(roi_data)
    
    print("Fit Complete.")
    
    # 4. Visualize
    print("Visualizing results...")
    # (Simple print for now)
    for k, v in fitted_params.items():
        print(f"{k}: Mean {np.mean(v)}")

if __name__ == "__main__":
    main()
