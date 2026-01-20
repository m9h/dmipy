
import time
import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.models.c_noddi import CNODDI
from dmipy_jax.fitting.optimization import OptimistixFitter

def main():
    print("================================================================")
    print("   DMIPY-JAX REAL DATA DEMO: STANFORD HARDI")
    print("================================================================")
    print(f"JAX Device: {jax.devices()[0]}")

    # 1. Fetch Data
    print("\n--> Fetching/Loading Stanford HARDI Dataset...")
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    
    data = img.get_fdata()
    affine = img.affine
    
    print(f"   Original Data Shape: {data.shape}")
    
    # 2. Preprocessing
    # Slice selection: Axial slice z=35
    z_slice = 35
    data_slice = data[:, :, z_slice, :]
    
    # Create Mask (Simple threshold on mean b0)
    # gtab.b0s_mask indicates b=0 indices
    b0_indices = np.where(gtab.b0s_mask)[0]
    mean_b0 = np.mean(data_slice[..., b0_indices], axis=-1)
    
    # Threshold for brain mask (heuristic)
    mask = mean_b0 > 300  
    
    # Flatten data for batch processing
    # Only select masked voxels to save time
    valid_voxels = data_slice[mask]
    n_voxels = valid_voxels.shape[0]
    
    print(f"   Processing Slice z={z_slice}")
    print(f"   Masked Voxels to Fit: {n_voxels}")
    
    # 3. Setup Acquisition
    # dmipy-jax expects bvals/bvecs. 
    # gtab.bvals and gtab.bvecs are available
    acq = JaxAcquisition(gtab.bvals, gtab.bvecs)
    
    # 4. Setup Model & Fitter
    print("\n--> Setting up CNODDI Fit...")
    model = CNODDI()
    
    # Parameter ranges: [theta, phi, f_stick, f_iso]
    ranges = [
        (0.0, np.pi),   
        (0.0, 2*np.pi), 
        (0.0, 1.0),     
        (0.0, 1.0)      
    ]
    # Scaling to keep params O(1)
    scales = jnp.array([1.0, 1.0, 1.0, 1.0])
    
    fitter = OptimistixFitter(model, ranges, scales=scales)
    
    # Compile fit function (vmap handled by VoxelFitter/OptimistixFitter internal if exposed?)
    # OptimistixFitter.fit is for single voxel.
    # We need to vmap it.
    
    # Define vmapped fit function
    # fit(data, acq, x0, sigma)
    # vmap over data (axis 0), acq (None), x0 (axis 0), sigma (None or axis 0)
    
    def fit_single(d, x0):
        # Estimates noise sigma roughly from data or constant
        # For simplicity in demo, assume constant SNR ~30 on b0 mean
        # sigma ~= mean_b0_voxel / 30.0
        # Actually let's just pass a rough scalar estimate 
        sigma = 10.0 
        return fitter.fit(d, acq, x0, sigma)
        
    fit_batch = jax.jit(jax.vmap(fit_single))
    
    # Initial guessing
    # Simple standardized guess for all voxels
    # [theta=1.5, phi=0, f_stick=0.5, f_iso=0.1]
    init_params = jnp.tile(jnp.array([1.5, 0.0, 0.5, 0.1]), (n_voxels, 1))
    
    # Run Fit
    print("--> Fitting...")
    t0 = time.time()
    # Convert input to JAX array
    data_jax = jnp.array(valid_voxels)
    
    fitted_params_batch, _ = fit_batch(data_jax, init_params)
    # block until ready
    fitted_params_batch.block_until_ready()
    t1 = time.time()
    
    duration = t1 - t0
    speed = n_voxels / duration
    print(f"   Fit Complete in {duration:.2f} s")
    print(f"   Processing Speed: {speed:.0f} voxels/sec")
    
    # 5. Reconstruct Maps
    print("\n--> Reconstructing Images...")
    # Initialize output array with zeros
    # Shape: (Nx, Ny, N_params)
    out_shape = mask.shape + (4,)
    out_map = np.zeros(out_shape)
    
    # Fill masked region
    # fitted_params_batch is (N_voxels, 4)
    # We need to assign it back. 
    # Can stick into flat array then reshape?
    # mask is (Nx, Ny).
    
    out_map[mask] = np.array(fitted_params_batch)
    
    # Valid Params: [theta, phi, f_stick, f_iso]
    f_stick_map = out_map[..., 2]
    f_iso_map = out_map[..., 3]
    
    # 6. Save NIfTI
    # Use empty header or original affine
    nib.save(nib.Nifti1Image(f_stick_map, affine), 'stanford_cnoddi_fstick.nii.gz')
    nib.save(nib.Nifti1Image(f_iso_map, affine), 'stanford_cnoddi_fiso.nii.gz')
    
    print(f"   Saved 'stanford_cnoddi_fstick.nii.gz'")
    print(f"   Saved 'stanford_cnoddi_fiso.nii.gz'")
    
    print("\n================================================================")
    print("DEMO COMPLETE")

if __name__ == "__main__":
    main()
