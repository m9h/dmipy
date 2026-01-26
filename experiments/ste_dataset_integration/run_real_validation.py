import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from experiments.ste_dataset_integration.ste_acquisition import STEAcquisitionScheme
from experiments.ste_dataset_integration.gradients import compute_spatial_gradient, compute_analytic_gradient

def load_data(subject_path):
    print(f"Loading data from {subject_path}")
    ste_path = subject_path / "STE" / "STE_degibbs_eddy.nii.gz"
    mask_path = subject_path / "STE" / "STE_mask.nii.gz"
    bvals_path = subject_path / "STE" / "bvals.txt"
    bvecs_path = subject_path / "STE" / "bvecs.txt"

    img = nib.load(ste_path)
    data = img.get_fdata()
    mask = nib.load(mask_path).get_fdata() > 0
    
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path)
    
    return data, mask, bvals, bvecs, img.header.get_zooms()[:3]

def real_geometry_validation():
    home = Path.home()
    subject_path = home / "Downloads/STE/STE01"
    
    if not subject_path.exists():
        print(f"Path not found: {subject_path}")
        return

    data, mask, bvals, bvecs, voxel_size = load_data(subject_path)
    print(f"Data shape: {data.shape}")
    print(f"Voxel size: {voxel_size}")
    
    # Preprocess: crop to a reasonable ROI to speed up and avoid empty background
    # Find center of mass of mask
    coords = np.argwhere(mask)
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)
    
    # Take a 40x40x40 chunk from center
    center = (coords.min(axis=0) + coords.max(axis=0)) // 2
    r = 20
    # ensure bounds
    sl_x = slice(center[0]-r, center[0]+r)
    sl_y = slice(center[1]-r, center[1]+r)
    sl_z = slice(center[2]-r, center[2]+r)
    
    roi_data = data[sl_x, sl_y, sl_z, :]
    roi_mask = mask[sl_x, sl_y, sl_z]
    
    # Select b=0 and b=1000
    b0_indices = np.where(bvals < 50)[0]
    b1000_indices = np.where(bvals > 50)[0]
    
    print(f"Found {len(b0_indices)} b0s and {len(b1000_indices)} b1000s")
    
    S0 = np.mean(roi_data[..., b0_indices], axis=-1)
    S_dwi = np.mean(roi_data[..., b1000_indices], axis=-1)
    
    # Compute ADC map (clip to avoid log(0) or neg)
    # ADC = -1/b * ln(S/S0)
    # b ~ 1000 s/mm^2. Let's work in s/mm^2 units for simplicity or convert.
    # Let's keep data as is.
    b_val = 1000.0
    
    # Avoid div by zero
    S0 = np.maximum(S0, 1e-6)
    ratio = S_dwi / S0
    ratio = np.clip(ratio, 1e-6, 1.0)
    
    ADC_map = -1.0/b_val * np.log(ratio)
    ADC_map = ADC_map * roi_mask # Zero out background
    
    # Convert to JAX
    ADC_jax = jnp.array(ADC_map)
    S0_jax = jnp.array(S0)
    voxel_size_jax = tuple(float(v) for v in voxel_size)
    
    # Calculate Gradients of Parameters (ADC and S0)
    # We are validating the Chain Rule: \nabla S = (\dS/dS0)\nabla S0 + (\dS/dADC)\nabla ADC
    
    grad_ADC = compute_spatial_gradient(ADC_jax, voxel_size=voxel_size_jax)
    grad_S0 = compute_spatial_gradient(S0_jax, voxel_size=voxel_size_jax)
    
    # Define Forward Model
    # S = S0 * exp(-b * ADC)
    # We need to simulate the signal for specific b-values.
    # Let's simulate for the actual Gradient Directions used in the scan? Use mean b=1000 for scalar ADC model.
    # To keep it rigorous, let's just simulate 3 synthetic measurements with varying b.
    
    sim_bvals = jnp.array([1000.0, 1000.0, 2000.0])
    # acquisition object (dummy for simple model)
    class SimpleAcq:
        bvalues = sim_bvals
    
    acq = SimpleAcq()
    
    def forward_model(params, acq):
        # params tuple: (S0, ADC)
        s0, adc = params
        return s0 * jnp.exp(-acq.bvalues * adc)

    # Predict Signal Volume
    # vmap over spatial dims
    # params structure must match what we pass to predicted
    # Let's stack params? Or keep as tuple?
    # vmap supports tupled args if we handle it.
    
    # Map over (S0_map, ADC_map)
    predict_scan = jax.vmap(jax.vmap(jax.vmap(
        lambda s, a: forward_model((s, a), acq)
    )))
    
    S_sim = predict_scan(S0_jax, ADC_jax) # (X, Y, Z, 3)
    
    # Compute Numerical Gradient of Signal
    grad_S_num = compute_spatial_gradient(S_sim, voxel_size=voxel_size_jax) # (3, X, Y, Z, 3)
    
    # Compute Analytic Gradient
    # We need to pass grad_S0 and grad_ADC as param_gradients list/tuple
    # Reshape grads for vmap (3, X, Y, Z) -> (X, Y, Z, 3) because we scan over X,Y,Z
    
    g_s0_in = jnp.moveaxis(grad_S0, 0, -1)
    g_adc_in = jnp.moveaxis(grad_ADC, 0, -1)
    
    analytic_scan = jax.vmap(jax.vmap(jax.vmap(
        lambda s, a, gs, ga: compute_analytic_gradient(
            forward_model, 
            (s, a), 
            (gs, ga), # gradients corresponding to params
            acq
        )
    )))
    
    grad_S_ana = analytic_scan(S0_jax, ADC_jax, g_s0_in, g_adc_in)
    # (X,Y,Z, 3, N_meas) -> (3, X, Y, Z, N_meas)
    grad_S_ana = jnp.moveaxis(grad_S_ana, 3, 0)
    
    # Compare
    # Mask out background defined by ROI mask (eroded to avoid edge effects)
    print(f"Grad S Ana Shape: {grad_S_ana.shape}")
    print(f"ROI Mask Shape: {roi_mask.shape}")
    
    # Erode by 1 voxel on all sides
    eroded_slice = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1), slice(None))
    mask_slice = (slice(1, -1), slice(1, -1), slice(1, -1))
    
    grad_eroded = grad_S_ana[eroded_slice]
    num_eroded = grad_S_num[eroded_slice]
    mask_eroded = roi_mask[mask_slice] > 0
    
    print(f"Eroded Grad Shape: {grad_eroded.shape}")
    print(f"Eroded Mask Shape: {mask_eroded.shape}")
    
    diff = num_eroded - grad_eroded
    
    # We only care about errors INSIDE the brain tissue where parameters are smooth-ish.
    # Filter by mask
    # Boolean indexing: diff[:, mask, :] matches axes 1,2,3
    # We need to flatten relevant axes manually or use where?
    # JAX bool indexing on multi-axis:
    # It prefers flat indexing or explicit taking.
    # Let's flatten spatial dims (X,Y,Z) -> M
    
    diff_flat = diff.reshape(3, -1, 3) # (3, XYZ, 3)
    grad_flat = grad_eroded.reshape(3, -1, 3)
    mask_flat = mask_eroded.flatten() # (XYZ,)
    
    diff_masked = diff_flat[:, mask_flat, :]
    sigs_masked = grad_flat[:, mask_flat, :]
    
    rel_err = jnp.linalg.norm(diff_masked) / (jnp.linalg.norm(sigs_masked) + 1e-9)
    
    print(f"Validation on Real Geometry (ROI size {roi_data.shape})")
    print(f"Relative Error (Norm): {rel_err:.2e}")
    
    # Real data has discontinuities at boundaries which might cause FD issues, 
    # but we are validating on the SIMULATED signal generated from the derived maps.
    # The derived maps (ADC) have sharp edges at the mask boundary.
    # FD will struggle with sharp edges.
    # But since we provide FD-computed parameter gradients to the analytic engine as input...
    # Wait, strict checking:
    # dS/dr = dS/dp * dp/dr
    # If we supply dp/dr = FD(p), then dS/dr should equal FD(S) ONLY if S(p) is linear?
    # No, S is nonlinear.
    # FD(S(p)) approx dS/dp * FD(p) + H.O.T.
    # If p jumps locally (edge), H.O.T are large.
    # Ideally we should smooth the parameter maps before test?
    # Let's apply a light gaussian smooth to S0 and ADC JAX arrays before running the test
    # to ensure the differentiation assumption holds well.
    
    from scipy.ndimage import gaussian_filter
    
    S0_smooth = jnp.array(gaussian_filter(S0, sigma=1.0))
    ADC_smooth = jnp.array(gaussian_filter(ADC_map, sigma=1.0))
    
    # Rerun with smoothed params
    print("--- Re-running with Smoothed Parameters (sigma=1.0) ---")
    
    # Grads
    g_s0_s = compute_spatial_gradient(S0_smooth, voxel_size=voxel_size_jax)
    g_adc_s = compute_spatial_gradient(ADC_smooth, voxel_size=voxel_size_jax)
    
    # Signal
    S_sim_s = predict_scan(S0_smooth, ADC_smooth)
    g_S_num_s = compute_spatial_gradient(S_sim_s, voxel_size=voxel_size_jax)
    
    # Analytic
    g_s0_in_s = jnp.moveaxis(g_s0_s, 0, -1)
    g_adc_in_s = jnp.moveaxis(g_adc_s, 0, -1)
    
    g_S_ana_s = analytic_scan(S0_smooth, ADC_smooth, g_s0_in_s, g_adc_in_s)
    g_S_ana_s = jnp.moveaxis(g_S_ana_s, 3, 0)
    
    grad_eroded_s = g_S_ana_s[eroded_slice]
    num_eroded_s = g_S_num_s[eroded_slice]
    
    diff_s = num_eroded_s - grad_eroded_s
    
    diff_flat_s = diff_s.reshape(3, -1, 3)
    grad_flat_s = grad_eroded_s.reshape(3, -1, 3)
    
    diff_masked_s = diff_flat_s[:, mask_flat, :]
    sigs_masked_s = grad_flat_s[:, mask_flat, :]
    
    rel_err_s = jnp.linalg.norm(diff_masked_s) / jnp.linalg.norm(sigs_masked_s)
    print(f"Relative Error (Smoothed): {rel_err_s:.2e}")
    
    # Relax tolerance for 2mm voxels where FD error is naturally higher
    assert rel_err_s < 0.10, f"Relative Error {rel_err_s:.2e} too high on real geometry."
    print("PASS: Real Geometry Gradient Validation successful.")

if __name__ == "__main__":
    real_geometry_validation()
