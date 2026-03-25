
import jax
# Enable x64 for precision in gradient checking
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import equinox as eqx
import sys
import os

# Add project root to path (assuming script run from root or examples/ste)
# To be safe, we add the current working directory if run from root.
sys.path.append(os.getcwd())

try:
    from dmipy_jax.acquisition import STEAcquisitionScheme
    from dmipy_jax.gradients.spatial import compute_spatial_gradient, predict_spatial_gradient
except ImportError:
    # Fallback if run inside examples/ste and root not in path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from dmipy_jax.acquisition import STEAcquisitionScheme
    from dmipy_jax.gradients.spatial import compute_spatial_gradient, predict_spatial_gradient

def synthetic_experiment():
    print("--- Starting Spatial Gradient Validation (Aganj et al. Logic) ---")
    
    # 1. Setup Acquisition
    # Create a simple protocol
    bvalues = jnp.array([0.0, 1000.0, 2000.0])
    # 3 Gradient directions
    gradient_directions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    mixing_time = 0.1 # seconds
    
    # We use SI units for b-values explicitly
    b_si = bvalues * 1e6 
    
    acq = STEAcquisitionScheme(
        bvalues=b_si, 
        gradient_directions=gradient_directions, 
        mixing_time=mixing_time
    )
    
    # 2. Define Synthetic Phantom (Parameters)
    # Grid size
    N = 50
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    z = jnp.linspace(0, 1, N)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Spatially varying Diffusivity D(r)
    # Simple pattern: Sinusoidal variation
    # D approx 1e-9 m^2/s
    D_base = 1e-9
    
    # D(x,y,z) variation
    D_map = D_base * (1.0 + 0.5 * jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y))
    D_map = D_map * (0.8 + 0.2 * jnp.cos(2 * jnp.pi * Z))
    
    # 3. Compute Numerical Parameter Gradients (Ground Truth for Input)
    voxel_size = 1.0 / (N - 1)
    grad_D = compute_spatial_gradient(D_map, voxel_size=voxel_size) # (3, X, Y, Z)
    
    # 4. Forward Model
    def mono_exponential_model(D, acq_scheme):
        # S = exp(-b * D)
        return jnp.exp(-acq_scheme.bvalues * D)

    # 5. Simulate Signal Volume
    # We vmap over X, Y, Z
    predict_volume = jax.vmap(jax.vmap(jax.vmap(
        lambda d: mono_exponential_model(d, acq)
    )))
    
    S_vol = predict_volume(D_map) # (X, Y, Z, N_meas)
    print(f"Signal Volume Shape: {S_vol.shape}")
    
    # 6. Compute Spatial Gradient of Signal (Finite Difference) -> "Observed" Gradient
    grad_S_num = compute_spatial_gradient(S_vol, voxel_size=voxel_size) # (3, X, Y, Z, N_meas)
    
    # 7. Compute Analytic Gradient (Prediction)
    # We predict dS/dr from dD/dr using chain rule.
    
    # Need to rearrange grad_D for vmap: (3, X, Y, Z) -> (X, Y, Z, 3)
    # Actually, predict_spatial_gradient expects param_gradients leaf to have shape (3, ...)
    # If D is scalar per voxel, D_map is (..., 1) effectively? No D_map is (X,Y,Z).
    # grad_D is (3, X, Y, Z).
    # We want to vmap over X, Y, Z.
    # So we move X,Y,Z to front.
    
    grad_D_vmap = jnp.moveaxis(grad_D, 0, -1) # (X, Y, Z, 3)
    # But wait, predict_spatial_gradient expects param_gradients leaf to start with 3.
    # No, predict_spatial_gradient is "single voxel".
    # At single voxel: D is scalar. grad_D is (3,).
    # So if we vmap, the inputs to the vmapped function should be:
    # D: scalar
    # grad_D: (3,)
    
    # So if D_map is (X,Y,Z) and grad_D is (3,X,Y,Z).
    # We want to vmap over axes 0,1,2 of D_map and axes 1,2,3 of grad_D?
    # Or simpler: rearrange grad_D to (X,Y,Z,3) and vmap over 0,1,2 for both.
    
    grad_D_perm = jnp.moveaxis(grad_D, 0, -1) # (X, Y, Z, 3)
    
    analytic_vmap = jax.vmap(jax.vmap(jax.vmap(
        lambda p, pg: predict_spatial_gradient(mono_exponential_model, p, pg, acq)
    )))
    
    grad_S_ana = analytic_vmap(D_map, grad_D_perm) 
    # Output of predict_spatial_gradient is (3, N)
    # Vmap output: (X, Y, Z, 3, N)
    
    # Rearrange to match grad_S_num: (3, X, Y, Z, N)
    grad_S_ana = jnp.moveaxis(grad_S_ana, 3, 0)
    
    # 8. Comparison
    # Crop borders (1 voxel) to avoid finite difference boundary errors
    valid_slice = np.s_[:, 1:-1, 1:-1, 1:-1, :]
    
    diff = grad_S_num[valid_slice] - grad_S_ana[valid_slice]
    err_subset = diff
    
    norm_diff = jnp.linalg.norm(err_subset)
    norm_gt = jnp.linalg.norm(grad_S_ana[valid_slice])
    rel_err = norm_diff / norm_gt
    
    max_abs_err = jnp.max(jnp.abs(err_subset))
    
    print(f"Relative Error (Norm): {rel_err:.2e}")
    print(f"Max Absolute Error: {max_abs_err:.2e}")
    
    # Tolerance: FD is O(h^2), h=0.02. Error ~ 4e-4. 
    # With double precision, we expect good match.
    assert rel_err < 5e-3, f"FAIL: Relative error {rel_err:.2e} too high (tol 5e-3)."
    print("PASS: Analytic vs Finite Difference Error within tolerance.")

if __name__ == "__main__":
    synthetic_experiment()
