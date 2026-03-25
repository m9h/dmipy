
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import os
import sys

# Add project root
sys.path.append(os.getcwd())

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.gradients.spatial import compute_spatial_gradient, predict_spatial_gradient

def showcase_mouse_ogse():
    print("--- Showcase: Spatial Gradients in OGSE Mouse Data (Synthetic Replica) ---")
    
    # 1. Define OGSE Protocols (Khan/Rahman et al. style)
    # Frequencies: 50, 100, 145, 190 Hz
    frequencies = jnp.array([50.0, 100.0, 145.0, 190.0])
    bval_mag = 800e6 # 800 s/mm^2 -> 800e6 s/m^2
    
    # Create simple acquisition items
    # We will treat each frequency as a separate "shell" for modeling
    # Protocol: 1 b0 + 4 freq shells * 1 direction
    # (Simplified 1D showcase for clarity)
    
    # In reality, OGSE models usually parameterize D(omega).
    # Model: D(omega) = D_inf + A / omega
    # S(omega) = exp(-b * D(omega))
    
    # 2. Create Phantom with Spatially Varying Disorder
    # Grid
    N = 64
    x = jnp.linspace(-1, 1, N)
    y = jnp.linspace(-1, 1, N)
    X, Y = jnp.meshgrid(x, y)
    
    # "Tumor" in the center with high disorder (strong frequency dependence)
    R = jnp.sqrt(X**2 + Y**2)
    tumor_mask = jnp.exp(-R**2 / 0.3)
    
    # Parameters map
    # D_inf: Bulk diffusivity (relatively constant)
    D_inf_map = 1.0e-9 * jnp.ones((N, N)) # 1.0 um^2/ms
    
    # A: Disorder parameter (High in tumor, Low in healthy tissue)
    # A determines how much D changes with frequency.
    # High A -> Slope of D(omega) is steep.
    A_map = 5.0e-9 * tumor_mask # Big change in center
    
    # 3. Predict Spatial Gradients from Parameters
    voxel_size = 2.0 / N # Field of view 2 units
    
    params_map = {'D_inf': D_inf_map, 'A': A_map}
    
    # Pad to 3D to satisfy compute_spatial_gradient (needs >=2 slices for z-gradient)
    # We duplicate slices
    depth = 3
    D_inf_map_3d = jnp.repeat(D_inf_map[..., None], depth, axis=2)
    A_map_3d = jnp.repeat(A_map[..., None], depth, axis=2)
    
    grad_D_inf = compute_spatial_gradient(D_inf_map_3d, voxel_size=(voxel_size, voxel_size, 1.0))
    grad_A = compute_spatial_gradient(A_map_3d, voxel_size=(voxel_size, voxel_size, 1.0))
    
    # compute_spatial_gradient returns (3, X, Y, Z)
    # Extract middle slice
    grad_D_inf = grad_D_inf[:, :, :, 1] # (3, X, Y)
    grad_A = grad_A[:, :, :, 1] # (3, X, Y)
    
    grad_params_map = {'D_inf': grad_D_inf, 'A': grad_A}
    
    # 4. Forward Model
    def ogse_model(p, acq):
        # acq has 'frequency' attribute? 
        # JaxAcquisition doesn't have frequency by default. 
        # We can pass frequency in 'delta' or 'Delta' or just make a custom one.
        # Or simpler: The "acquisition" argument in predict_spatial_gradient is passed to the model.
        # We can pass the frequency array directly as the "acquisition" if the model expects it,
        # OR better, standard way:
        
        freq = acq.echo_time # HACK: Store freq in echo_time for this demo
        b = acq.bvalues
        
        d_omega = p['D_inf'] + p['A'] / (freq + 1e-9)
        return jnp.exp(-b * d_omega)

    # 5. Simulate for one specific frequency shell (e.g. 50 Hz)
    target_freq = 50.0
    
    # Make "Acquisition" object
    acq_50 = JaxAcquisition(
        bvalues=jnp.array([bval_mag]),
        gradient_directions=jnp.array([[1.0, 0.0, 0.0]]),
        echo_time=jnp.array([target_freq]) # Storing freq here
    )
    
    # Predict Signal Volume
    # vmap over X, Y
    simulate_vol = jax.vmap(jax.vmap(
        lambda p: ogse_model(p, acq_50)
    ))
    
    S_vol_50 = simulate_vol(params_map) # (N, N, 1)
    S_vol_50 = S_vol_50[..., 0] # (N, N)
    
    # 6. Predict Gradient (Analytic)
    # vmap over X, Y
    # Inputs: params (scalar), grad_params (3,)
    # We need to rearrange inputs for vmap
    
    # params_map: {'D_inf': (X,Y), 'A': (X,Y)}
    # grad_params_map: {'D_inf': (3,X,Y), 'A': (3,X,Y)}
    
    # Move spatial axes to front for vmap?
    # Actually vmap iterates over leading axis.
    # params_map has leading axis X.
    
    # grad_params_map has leading axis 3. We need it to be (X, Y, 3).
    grad_params_perm = {
        'D_inf': jnp.moveaxis(grad_D_inf, 0, -1),
        'A': jnp.moveaxis(grad_A, 0, -1)
    }
    
    predict_AD = jax.vmap(jax.vmap(
         lambda p, pg: predict_spatial_gradient(ogse_model, p, pg, acq_50)
    ))
    
    grad_S_ana = predict_AD(params_map, grad_params_perm) 
    # Output: (X, Y, 3, 1) -> (X, Y, 3)
    grad_S_ana = grad_S_ana[..., 0]
    
    # 7. Compute Numerical Gradient
    # S_vol_50 shape (X, Y)
    # Pad to 3D for gradient calculation
    S_vol_3d = jnp.repeat(S_vol_50[..., None], 3, axis=2)
    
    grad_S_num = compute_spatial_gradient(S_vol_3d, voxel_size=(voxel_size, voxel_size, 1.0))
    # Shape (3, X, Y, Z) -> extract middle slice, Z index 1
    grad_S_num = grad_S_num[:, :, :, 1]
    
    # Compare
    # Move ana to (3, X, Y)
    grad_S_ana = jnp.moveaxis(grad_S_ana, 2, 0)
    
    # Slice valid region
    diff = grad_S_ana[:, 1:-1, 1:-1] - grad_S_num[:, 1:-1, 1:-1]
    rel_err = jnp.linalg.norm(diff) / jnp.linalg.norm(grad_S_num[:, 1:-1, 1:-1])
    
    print(f"Frequency: {target_freq} Hz")
    print(f"Max Signal: {jnp.max(S_vol_50):.4f}")
    print(f"Gradient Match Error (Rel): {rel_err:.2e}")
    
    # Save Image
    # Create NIfTI
    # Use nibabel directly
    
    affine = jnp.eye(4) * voxel_size
    affine = affine.at[3,3].set(1.0)
    
    img = nib.Nifti1Image(np.array(S_vol_50[:, :, None]), np.array(affine))
    nib.save(img, "examples/ste/mouse_ogse_50hz_sim.nii.gz")
    print("Saved simulation to examples/ste/mouse_ogse_50hz_sim.nii.gz")
    
    assert rel_err < 1e-2, "Gradient mismatch too high!"
    print("SUCCESS: Spatial gradient prediction works on frequency-dependent OGSE model.")

if __name__ == "__main__":
    showcase_mouse_ogse()
