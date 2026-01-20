
"""
Advanced Processing Pipeline for Connectome 2.0 (ds006181).
Leverages 'dmipy-jax' high-performance tools:
- RestrictedCylinder model (Soderman/Callaghan) for High-G sensitivity.
- OptimistixFitter for robust non-linear optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dmipy_jax.io.connectome2 import load_connectome2_mri
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models, zeppelin
from dmipy_jax.fitting.optimization import OptimistixFitter

def main():
    print("--- Connectome 2.0 Advanced Processing ---")
    
    # 1. Load Data
    print("Loading data...")
    # New signature from user refactor
    data_dict = load_connectome2_mri() 
    data = data_dict['dwi']
    scheme = data_dict['scheme']
    
    print(f"Data Shape: {data.shape}")
    
    # Crop to ROI
    mid_x, mid_y, mid_z = np.array(data.shape[:3]) // 2
    # Expanding ROI for better visualization (axially)
    roi_slice = (
        slice(mid_x-20, mid_x+20), 
        slice(mid_y-20, mid_y+20), 
        slice(mid_z, mid_z+1)
    )
    roi_data = data[roi_slice]
    print(f"ROI Shape: {roi_data.shape}")
    
    # 2. Define Advanced Model
    # Connectome 2.0 is ideal for measuring axon diameter.
    # We use RestrictedCylinder (Cylinder with finite radius)
    # Note: This requires Delta/delta to be present in the scheme.
    
    print("Defining Model: RestrictedCylinder + Zeppelin + Ball")
    # Restricted Cylinder: Intra-axonal
    # Connectome 2.0 High-G data allows estimation of axon diameter.
    intra = cylinder_models.RestrictedCylinder(diameter=None) # Allow fitting diameter
    extra = zeppelin.Zeppelin()
    csf = gaussian_models.Ball()
    
    model = JaxMultiCompartmentModel(models=[intra, extra, csf])
    
    # 3. Fit
    print("Fitting with OptimistixFitter...")
    fitter = OptimistixFitter(model, scheme)
    fitted_params = fitter.fit(roi_data)
    
    print("Fit Complete.")
    
    # 4. Visualize
    print("Saving visualization to connectome2_maps.png...")
    
    # keys: 'RestrictedCylinder_1_diameter', etc.
    # Let's find the diameter key
    diam_key = next((k for k in fitted_params.keys() if 'diameter' in k), None)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Param 1: Diameter
    if diam_key:
        im0 = axes[0].imshow(fitted_params[diam_key][:,:,0], cmap='magma', origin='lower')
        axes[0].set_title(f"Axon Diameter\n{diam_key}")
        plt.colorbar(im0, ax=axes[0])
    
    # Param 2: Intra Volume Fraction
    # Parameter names are usually model_name_1_partial_volume_0 or similar
    # We inspect keys
    vf_intra_key = next((k for k in fitted_params.keys() if 'RestrictedCylinder' in k and 'volume' in k), None)
    if vf_intra_key:
        im1 = axes[1].imshow(fitted_params[vf_intra_key][:,:,0], cmap='viridis', vmin=0, vmax=1, origin='lower')
        axes[1].set_title("Intra-Axonal VF")
        plt.colorbar(im1, ax=axes[1])
        
    # Param 3: Extra Volume Fraction
    vf_extra_key = next((k for k in fitted_params.keys() if 'Zeppelin' in k and 'volume' in k), None)
    if vf_extra_key:
        im2 = axes[2].imshow(fitted_params[vf_extra_key][:,:,0], cmap='viridis', vmin=0, vmax=1, origin='lower')
        axes[2].set_title("Extra-Axonal VF")
        plt.colorbar(im2, ax=axes[2])
        
    # Param 4: CSF Volume Fraction
    vf_csf_key = next((k for k in fitted_params.keys() if 'Ball' in k and 'volume' in k), None)
    if vf_csf_key:
        im3 = axes[3].imshow(fitted_params[vf_csf_key][:,:,0], cmap='bone', vmin=0, vmax=1, origin='lower')
        axes[3].set_title("CSF VF")
        plt.colorbar(im3, ax=axes[3])
        
    plt.tight_layout()
    plt.savefig("connectome2_maps.png")
    print("Saved connectome2_maps.png")

if __name__ == "__main__":
    main()
