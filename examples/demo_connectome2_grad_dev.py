"""
Demonstration of Gradient Nonlinearity Correction for Connectome 2.0.

This script demonstrates:
1. Loading Connectome 2.0 data (or synthesizing it if not present).
2. Loading/Synthesizing a Gradient Nonlinearity Tensor (grad_dev).
3. Applying the gradient nonlinearity to the acquisition scheme (B-value inhomogeneity).
4. Fitting the data using `dmipy-jax` with spatially varying acquisition.
5. Comparing "Standard" (uncorrected) vs "Corrected" results.

Objective: Flatten the b-value inhomogeneity and recover accurate microstructure.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dmipy_jax.io.connectome2 import load_connectome2_mri, load_gradient_nonlinearity, apply_gradient_nonlinearity
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models, zeppelin
from dmipy_jax.fitting.optimization import OptimistixFitter
from dmipy_jax.acquisition import JaxAcquisition

def synthesize_data_and_grad_dev(shape=(20, 20, 1)):
    """
    Synthesizes dummy data and a gradient nonlinearity tensor for demonstration 
    if the real dataset is not available.
    """
    print("Synthesizing dummy Connectome 2.0 data...")
    N_vox = np.prod(shape)
    
    # 1. Nominal Scheme (High Field, Strong Gradients)
    # 3 shells: 1000, 3000, 10000 s/mm^2
    bvals = np.concatenate([np.ones(10)*1000, np.ones(10)*3000, np.ones(10)*10000]) * 1e6 # SI units
    # Random directions
    from dipy.core.geometry import sphere2cart
    theta = np.random.rand(30) * np.pi
    phi = np.random.rand(30) * 2 * np.pi
    x, y, z = sphere2cart(1, theta, phi)
    bvecs = np.stack([x, y, z], axis=1)
    
    # Add timings (needed for RestrictedCylinder)
    delta = 0.012 # 12ms
    Delta = 0.035 # 35ms
    nominal_scheme = JaxAcquisition(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=delta,
        Delta=Delta
    )
    
    # 2. Synthesize Gradient Nonlinearity Tensor (Spatial Variation)
    # L(r) = I + E(r), where E is small error
    # We create a "warp" that increases radially from center
    X, Y, Z = np.meshgrid(
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[0]),
        np.linspace(0, 0, shape[2])
    )
    
    # Error tensor field (N_vox, 3, 3)
    # Scale error by distance from center
    R = np.sqrt(X**2 + Y**2)
    
    # Simple dilation/compression:
    # L = [[1+c*R, 0, 0], [0, 1+c*R, 0], [0, 0, 1]]
    c = 0.2 # 20% max deviation at edge
    
    L = np.zeros(shape + (3, 3))
    L[..., 0, 0] = 1.0 + c * R
    L[..., 1, 1] = 1.0 + c * R
    L[..., 2, 2] = 1.0 # Z is constant here
    
    # 3. Simulate Signal using TRUE local gradients
    # True Microstructure: Constant Cylinder Diameter = 3.0 microns
    true_diam = 3.0e-6
    intra = cylinder_models.RestrictedCylinder(diameter=None)
    extra = zeppelin.Zeppelin()
    csf = gaussian_models.Ball()
    model = JaxMultiCompartmentModel([intra, extra, csf])
    
    # Ground Truth Params (Constant across image)
    # Ground Truth Params (Constant across image)
    # Parameter names based on collision logic:
    # 0 (Cylinder): 'mu', 'diameter' (if diameter fits)
    # 1 (Zeppelin): 'mu' -> 'mu_2', 'lambda_par', 'lambda_perp'
    # 2 (Ball): 'lambda_iso'
    # Partial Volumes: 'partial_volume_0', 'partial_volume_1', 'partial_volume_2'
    
    gt_params = {
        'diameter': jnp.ones(N_vox) * true_diam,
        'mu': jnp.tile(jnp.array([0., 0.]), (N_vox, 1)), # Cylinder orientation
        'lambda_par': jnp.ones(N_vox) * 1.7e-9,          # Cylinder parallel diffusivity
        
        'mu_2': jnp.tile(jnp.array([0., 0.]), (N_vox, 1)), # Zeppelin orientation
        'lambda_par_2': jnp.ones(N_vox) * 1.7e-9,          # Zeppelin parallel diffusivity
        'lambda_perp': jnp.ones(N_vox) * 0.5e-9,           # Zeppelin perpendicular diffusivity
        
        'lambda_iso': jnp.ones(N_vox) * 3.0e-9,            # CSF
        'partial_volume_0': jnp.ones(N_vox) * 0.6, # intra
        'partial_volume_1': jnp.ones(N_vox) * 0.3, # extra
        'partial_volume_2': jnp.ones(N_vox) * 0.1  # csf
    }
    
    # Create Local Scheme
    grad_dev_tensor = jnp.array(L)
    local_scheme = apply_gradient_nonlinearity(nominal_scheme, grad_dev_tensor)
    
    # Simulate
    # JaxMultiCompartmentModel expects batched inputs if scheme is batched?
    # Our update allows (0, 0, 0)
    
    # We need to flatten gt_params dictionary arrays for vmap?
    # No, __call__ handles dictionary of arrays.
    
    print("Simulating ground truth signal with GNL...")
    data_flat = model(gt_params, local_scheme) # (N_vox, N_meas)
    data = data_flat.reshape(shape + (-1,))
    
    return data, nominal_scheme, grad_dev_tensor, gt_params

def main():
    print("--- Connectome 2.0 Gradient Nonlinearity Demo ---")
    
    # 1. Load or Synthesize Data
    try:
        print("Attempting to load real Connectome 2.0 data...")
        data_dict = load_connectome2_mri() # Default subject
        data = data_dict['dwi']
        scheme = data_dict['scheme']
        grad_dev = load_gradient_nonlinearity(data_dict['dataset_path'], data_dict['subject'])
        
        # Crop to ROI for speed
        mid_x, mid_y, mid_z = np.array(data.shape[:3]) // 2
        roi_slice = (slice(mid_x-20, mid_x+20), slice(mid_y-20, mid_y+20), slice(mid_z, mid_z+1))
        data = data[roi_slice]
        
        if grad_dev is None:
            raise FileNotFoundError("Real grad_dev not found")
            
        grad_dev = grad_dev[roi_slice]
        print(f"Loaded Real Data ROI: {data.shape}")
        
    except (FileNotFoundError, IndexError, Exception) as e:
        print(f"Could not load real data ({e}). Using Synthetic Data.")
        data, scheme, grad_dev, gt_params = synthesize_data_and_grad_dev(shape=(40, 40, 1))

    # 2. Prepare Schemes
    print("Preparing acquisition schemes...")
    
    # A. Nominal Scheme (Uncorrected)
    # Using 'scheme' directly. It is static (N_meas,)
    
    # B. Corrected Scheme (Batched)
    print("Applying Gradient Nonlinearity Tensor...")
    corrected_scheme = apply_gradient_nonlinearity(scheme, grad_dev)
    
    # Visualize B-value Inhomogeneity
    # Take max b-value shell
    max_b_idx = jnp.argmax(scheme.bvalues)
    # On the map, show b-value of this measurement across space
    b_map = corrected_scheme.bvalues[:, max_b_idx]
    b_map_img = b_map.reshape(data.shape[:-1])
    
    # 3. Define Model
    # Restricted Cylinder to estimate Diameter
    intra = cylinder_models.RestrictedCylinder(diameter=None)
    extra = zeppelin.Zeppelin()
    csf = gaussian_models.Ball()
    model = JaxMultiCompartmentModel([intra, extra, csf])
    
    # 4. Fit 1: Standard (Uncorrected)
    print("\n--- Fitting Standard (Uncorrected) ---")
    # model.fit returns the dictionary of parameters directly
    params_std = model.fit(scheme, data)
    
    # 5. Fit 2: Corrected
    print("\n--- Fitting Corrected (With GNL) ---")
    # We pass the BATCHED corrected_scheme
    # The updated fitter detects batched scheme and switches to vmap(0, 0, 0)
    params_cor = model.fit(corrected_scheme, data)
    
    # 6. Visualize Results
    print("Plotting results...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Context
    # B-value Map
    im0 = axes[0,0].imshow(b_map_img, cmap='jet')
    axes[0,0].set_title(f"Actual B-value Map\nNominal: {scheme.bvalues[max_b_idx]:.0f}")
    plt.colorbar(im0, ax=axes[0,0])
    
    # Diameter Standard
    diam_key = 'diameter'
    # Check key exist (just in case I guessed wrong, but I'm confident)
    if diam_key not in params_std:
        # Fallback search
        diam_key = [k for k in params_std.keys() if 'diameter' in k][0]
        print(f"DEBUG: Found diameter key: {diam_key}")
        
    d_std = params_std[diam_key].reshape(data.shape[:-1]) * 1e6 # microns
    im1 = axes[0,1].imshow(d_std, cmap='magma', vmin=0, vmax=10)
    axes[0,1].set_title("Est. Diameter (Standard)\nUncorrected")
    plt.colorbar(im1, ax=axes[0,1])
    
    # Diameter Corrected
    d_cor = params_cor[diam_key].reshape(data.shape[:-1]) * 1e6
    im2 = axes[0,2].imshow(d_cor, cmap='magma', vmin=0, vmax=10)
    axes[0,2].set_title("Est. Diameter (Corrected)\nWith Grad Dev")
    plt.colorbar(im2, ax=axes[0,2])
    
    # Row 2: Error Analysis (if synthetic)
    # Or just subtraction
    diff = d_std - d_cor
    im3 = axes[1,1].imshow(diff, cmap='coolwarm', vmin=-2, vmax=2)
    axes[1,1].set_title("Difference (Std - Cor)")
    plt.colorbar(im3, ax=axes[1,1])
    
    # Histogram
    axes[1,2].hist(d_std.flatten(), bins=30, alpha=0.5, label='Standard')
    axes[1,2].hist(d_cor.flatten(), bins=30, alpha=0.5, label='Corrected')
    axes[1,2].legend()
    axes[1,2].set_title("Diameter Distribution")
    
    # Text
    axes[1,0].axis('off')
    info = (
        "Demonstration:\n"
        "Connectome 2.0 strong gradients cause B-value inhomogeneity.\n"
        "Without correction, fitted parameters (e.g. diameter) vary spatially\n"
        "even if tissue is homogeneous.\n"
        "Using `grad_dev` flattens the result."
    )
    axes[1,0].text(0.1, 0.5, info, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("grad_dev_demo_results.png")
    print("Saved grad_dev_demo_results.png")

if __name__ == "__main__":
    main()
