
import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.io.connectome2 import load_connectome2_mri
from dmipy_jax.signal_models.sandi import get_sandi_model
from dmipy_jax.acquisition import JaxAcquisition

def run_sandi():
    print("Initializing SANDI (Connectome 2.0) Example...")
    
    # 1. Load Data
    # Only load a single slice to save time/memory for the example
    print("Loading Connectome 2.0 data (sub-01)...")
    try:
        data_dict = load_connectome2_mri(subject="sub-01", voxel_slice=(slice(40, 60), slice(40, 60), 30))
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Please ensure you have datalad installed and the dataset fetched.")
        return

    data = data_dict['dwi']
    bvals = data_dict['bvals']
    bvecs = data_dict['bvecs']
    
    print(f"Loaded data shape: {data.shape}")
    print(f"B-values range: {jnp.min(bvals):.2f} - {jnp.max(bvals):.2f}")
    
    # 2. Define Acquisition with Timing
    # SANDI requires delta and Delta.
    # Heuristic for Connectome 2.0 (based on preliminary checks):
    # High b-values usually imply longer diffusion times or high G.
    # We will approximate standard Connectom protocol values if exact metadata is missing.
    # delta = 12.9 ms, Delta = 21.8 ms (Standard HCP/Connectom intermediate)
    # OR the high-grad specific: delta=8ms, Delta=20ms?
    # Let's use a reasonable fixed timing for demonstration if we can't parse it.
    
    # But wait, Palombo et al 2020 emphasizes the importance of G.
    # G = sqrt(b / (delta^2 (Delta - delta/3))) * gamma_inv
    # If we get timing wrong, our estimated radius will be biased.
    
    # Let's assume standard parameters:
    # delta = 0.0129 s
    # Delta = 0.0218 s
    delta_est = 0.0129
    Delta_est = 0.0218
    
    acquisition = JaxAcquisition(
        bvalues=jnp.array(bvals),
        gradient_directions=jnp.array(bvecs),
        delta=delta_est,
        Delta=Delta_est
    )
    
    # 3. Define SANDI Model
    print("Defining SANDI Model...")
    # Stick + Sphere + Zeppelin + Ball
    # We use the helper
    sandi_func = get_sandi_model(
        sphere_diameter_range=(2e-6, 15e-6)
    )
    
    # 4. Fitting
    # Define Loss
    @jax.jit
    def loss_fn(params, signal_target, acq):
        # Params: [theta, phi, f_stick, f_sphere, f_ball, diameter, lambda_perp]
        # Constraints applied via transformation or clipping inside?
        # The model function expects raw params but we should constrain them.
        
        # We'll expect params in unconstrained space and sigmoid them here?
        # Or just clip. Simpler to clip for this demo.
        
        # Fractions sum to <= 1
        # We optimize f_stick, f_sphere, f_ball directly.
        
        pred = sandi_func(params, acq)
        return jnp.mean((pred - signal_target)**2)
        
    # Optimizer
    optimizer = optax.adam(learning_rate=0.01)
    
    # ROI Fitting
    # Flatten ROI
    n_x, n_y, n_dwis = data.shape
    data_flat = data.reshape(-1, n_dwis)
    
    # Normalize
    b0_mask = bvals < 50
    b0_mean = jnp.mean(data_flat[:, b0_mask], axis=1, keepdims=True)
    b0_mean = jnp.maximum(b0_mean, 1e-6) # Avoid div zero
    data_norm = data_flat / b0_mean
    
    print(f"Fitting {data_norm.shape[0]} voxels...")
    
    # Initial Guess
    # theta=0, phi=0, f_stick=0.3, f_sphere=0.3, f_ball=0.1, D=8um, perp=0.5
    init_params = jnp.array([0.0, 0.0, 0.3, 0.3, 0.1, 8.0e-6, 0.5e-9])
    
    # Vmap the optimization step? 
    # For 400 voxels, a loop is fine, or simple vmap.
    
    @jax.jit
    def fit_voxel(signal, init_p):
        opt_state = optimizer.init(init_p)
        
        def step(carry, i):
            params, state = carry
            l, grads = jax.value_and_grad(loss_fn)(params, signal, acquisition)
            updates, state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Constraints
            # Fractions [0, 1]
            new_params = new_params.at[2].set(jnp.clip(new_params[2], 0.01, 0.9)) # stick
            new_params = new_params.at[3].set(jnp.clip(new_params[3], 0.01, 0.9)) # sphere
            new_params = new_params.at[4].set(jnp.clip(new_params[4], 0.01, 0.9)) # ball
            
            # Normalize sum? If sum > 1, scale down?
            # sum_f = new_params[2] + new_params[3] + new_params[4]
            # scale = jnp.maximum(sum_f, 1.0)
            # new_params = new_params.at[2:5].set(new_params[2:5] / scale) 
            # (Simple clip is often enough if regularization pulls them down, but let's leave it flexible)
            
            # Diameter [1um, 20um]
            new_params = new_params.at[5].set(jnp.clip(new_params[5], 1e-6, 20e-6))
            
            # Lambda perp [0, 3e-9]
            new_params = new_params.at[6].set(jnp.clip(new_params[6], 0.0, 3e-9))
            
            return (new_params, state), l
            
        (final_params, _), _ = jax.lax.scan(step, (init_p, opt_state), jnp.arange(300))
        return final_params
    
    # Run fit
    # Using vmap to fit all voxels in parallel
    fitted_params = jax.vmap(fit_voxel, in_axes=(0, None))(data_norm, init_params)
    
    print("Fitting complete.")
    
    # Reshape results
    fitted_map = fitted_params.reshape(n_x, n_y, -1)
    
    # 5. Visualization
    f_sphere_map = fitted_map[..., 3]
    diameter_map = fitted_map[..., 5] * 1e6 # um
    
    print(f"Mean Soma Fraction: {jnp.mean(f_sphere_map):.3f}")
    print(f"Mean Diameter: {jnp.mean(diameter_map):.2f} um")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(jnp.rot90(f_sphere_map), cmap='hot', vmin=0, vmax=1)
    plt.title("Soma Fraction (f_sphere)")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.rot90(diameter_map), cmap='viridis', vmin=0, vmax=20)
    plt.title("Soma Diameter (um)")
    plt.colorbar()
    
    out_path = "sandi_connectome_results.png"
    plt.savefig(out_path)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_sandi()
