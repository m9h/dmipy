import jax
# import jax.config
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import equinox as eqx
import sys
import os

# Add project root to path to allow imports from experiments
sys.path.append(os.getcwd())

from experiments.ste_dataset_integration.ste_acquisition import STEAcquisitionScheme
from experiments.ste_dataset_integration.gradients import compute_spatial_gradient, compute_analytic_gradient

def synthetic_experiment():
    print("--- Starting Aganj Test (Spatial Gradient Validation) ---")
    
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
    
    acq = STEAcquisitionScheme(
        bvalues=bvalues, 
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
    D0 = 1e-9 
    b_si = bvalues * 1e6 
    
    # Update acq
    acq = STEAcquisitionScheme(
        bvalues=b_si,
        gradient_directions=gradient_directions,
        mixing_time=mixing_time
    )
    
    D_map = 1e-9 * (1.0 + 0.5 * jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y))
    D_map = D_map * (0.8 + 0.2 * jnp.cos(2 * jnp.pi * Z))
    
    # 3. Compute Numerical Parameter Gradients
    voxel_size = 1.0 / (N - 1)
    grad_D = compute_spatial_gradient(D_map, voxel_size=voxel_size) # (3, X, Y, Z)
    
    # 4. Forward Model
    def mono_exponential_model(D, acq_scheme):
        return jnp.exp(-acq_scheme.bvalues * D)

    # 5. Simulate Signal Volume
    predict_volume = jax.vmap(jax.vmap(jax.vmap(
        lambda d: mono_exponential_model(d, acq)
    )))
    
    S_vol = predict_volume(D_map) # (X, Y, Z, N_meas)
    print(f"Signal Volume Shape: {S_vol.shape}")
    
    # 6. Compute Spatial Gradient of Signal (Finite Difference)
    grad_S_num = compute_spatial_gradient(S_vol, voxel_size=voxel_size) # (3, X, Y, Z, N_meas)
    
    # 7. Compute Analytic Gradient
    grad_D_vmap = jnp.moveaxis(grad_D, 0, -1)
    
    analytic_vmap = jax.vmap(jax.vmap(jax.vmap(
        lambda p, pg: compute_analytic_gradient(mono_exponential_model, p, pg, acq)
    )))
    
    grad_S_ana = analytic_vmap(D_map, grad_D_vmap)
    # Transpose back: (X, Y, Z, 3, N) -> (3, X, Y, Z, N)
    grad_S_ana = jnp.moveaxis(grad_S_ana, 3, 0)
    
    # 8. Comparison
    # Crop borders (1 voxel)
    valid_slice = np.s_[:, 1:-1, 1:-1, 1:-1, :]
    
    diff = grad_S_num - grad_S_ana
    err_subset = diff[valid_slice]
    rel_err = jnp.linalg.norm(err_subset) / jnp.linalg.norm(grad_S_ana[valid_slice])
    
    print(f"Relative Error (Norm): {rel_err:.2e}")
    print(f"Max Absolute Error: {jnp.max(jnp.abs(err_subset)):.2e}")
    
    assert rel_err < 5e-3, f"FAIL: Relative error {rel_err:.2e} too high (tol 5e-3)."
    print("PASS: Analytic vs Finite Difference Error within tolerance.")

if __name__ == "__main__":
    synthetic_experiment()
