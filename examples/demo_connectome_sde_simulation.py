
"""
Data-Driven SDE Simulation using Connectome 2.0 (ds006181).

This script demonstrates "Biologically Plausible Simulation" by:
1. Loading real Connectome 2.0 DWI data.
2. Fitting a DTI model to extract:
   - Primary Eigenvector (V1) -> Becomes the **Vector Field** (T).
   - Fractional Anisotropy (FA) -> Inverted to create the **Confinement Potential** (U).
3. Simulating particle diffusion using `CurvedTractSDE`.
   - Particles should flow along the real white matter pathways.
   - Particles should be confined to high-FA regions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# dmipy-jax imports
from dmipy_jax.io.connectome2 import load_connectome2_mri
from dmipy_jax.simulation.sde_models import CurvedTractSDE, solve_restricted_sde_batch
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table

def main():
    print("--- Connectome 2.0 Data-Driven SDE Simulation ---")
    
    # 1. Load Real Data
    print("Loading Connectome 2.0 data subset...")
    # Loading a smaller ROI to keep it fast
    # Center of brain usually has Corpus Callosum (high FA)
    # Let's try to load a central slice
    try:
        data_dict = load_connectome2_mri()
        full_data = data_dict['dwi']
        affine = data_dict['affine']
        bvals = data_dict['bvals']
        bvecs = data_dict['bvecs']
        
        # Define ROI (Central Slice, Corpus Callosum area)
        # Shape is usually (140, 140, 90) or similar for 1.5mm
        shape = full_data.shape[:3]
        mx, my, mz = shape[0]//2, shape[1]//2, shape[2]//2
        
        # Roi: 40x40 patch, 5 slices thick
        roi_slice = (
            slice(mx-20, mx+20),
            slice(my-20, my+20),
            slice(mz-2, mz+3)
        )
        data = full_data[roi_slice] 
        print(f"ROI Data Shape: {data.shape}")
        
    except Exception as e:
        print(f"Failed to load real C2.0 data: {e}")
        print("Falling back to Synthetic Phantom for demonstration.")
        # Create synthetic V1/FA
        S = (40, 40, 5)
        # Fake CC (X-oriented)
        V1 = np.zeros(S + (3,))
        V1[:, :, :, 0] = 1.0 # X-dir
        # Fake FA (Band in middle)
        FA = np.zeros(S)
        FA[10:30, :, :] = 0.8 # High FA band
        affine = np.eye(4)
        data = None # Skip DTI fit
    
    # 2. Fit DTI to get Fields (if data loaded)
    if data is not None:
        print("Fitting DTI to extract Vector Field (V1) and Potential (FA)...")
        gtab = gradient_table(bvals, bvecs)
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(np.array(data))
        
        FA = tenfit.fa
        V1 = tenfit.evecs[..., 0] # Primary eigenvector (H,W,D,3)
        
        # Clean NaNs
        FA = np.nan_to_num(FA)
        V1 = np.nan_to_num(V1)
        
    # 3. Construct Fields for SDE
    # Vector Field: V1 (Orientation)
    # Potential Field: U = 1 - FA (Low potential in WM, High in GM/CSF)
    # This acts as a "Riverbed": Particles flow in WM (FA~1, U~0) and bounce off walls (FA~0, U~1).
    
    vector_field_jax = jnp.array(V1.transpose(3, 0, 1, 2)) # (3, H, W, D)
    
    # Sharpen potential to make walls harder?
    # U = (1 - FA)^2 * Strength
    potential_base = 1.0 - FA
    potential_field_jax = jnp.array(potential_base)
    
    print("Fields prepared.")
    print(f"Vector Field range: {vector_field_jax.min()} to {vector_field_jax.max()}")
    print(f"Potential Field range: {potential_field_jax.min()} to {potential_field_jax.max()}")
    
    # 4. Initialize SDE Model
    # We use a JAX affine that matches the ROI crop if needed, but here we just use Identity
    # and map voxel coords 1:1 for simplicity in this demo.
    model_affine = jnp.eye(4) 
    
    sde_model = CurvedTractSDE(
        vector_field=vector_field_jax,
        potential_field=potential_field_jax,
        affine=model_affine,
        diffusivity_long=1.5,   # um^2/ms (Fast along axon)
        diffusivity_trans=0.1,  # um^2/ms (Slow perp)
        k_confinement=20.0      # Strong push towards High FA
    )
    
    # 5. Simulation Setup
    # Seed particles in High FA regions
    print("Seeding particles in High FA regions...")
    
    # Find indices where FA > 0.6
    seed_indices = jnp.argwhere(jnp.array(FA) > 0.6)
    if len(seed_indices) == 0:
        print("No high FA regions found! Seeding randomly.")
        # Fallback
        H, W, D = FA.shape
        seed_indices = jnp.array([[H//2, W//2, D//2]])

    # Pick N random seeds from valid spots
    N_particles = 200
    rng = np.random.default_rng(42)
    choice = rng.choice(len(seed_indices), N_particles, replace=True)
    seeds_voxel = seed_indices[choice].astype(float)
    
    # JAX inputs
    y0 = jnp.array(seeds_voxel)
    key = jax.random.PRNGKey(1337)
    
    print(f"Simulating {N_particles} particles...")
    T_max = 50.0 # ms
    dt = 0.05
    
    sol = solve_restricted_sde_batch(sde_model, (0, T_max), y0, dt0=dt, key=key)
    
    final_pos = sol.ys[:, -1, :]
    
    
    # 6. Signal Generation
    print("Computing dMRI Signals from Trajectories...")
    # Define Acquisition Scheme (Simple 3-shell subset of C2.0)
    # G=80mT/m, 150mT/m, 300mT/m. 
    # Delta=30ms, delta=12ms.
    directions = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.707, 0.707, 0], [0.707, 0, 0.707], [0, 0.707, 0.707],
        [-0.707, 0.707, 0], [0.577, 0.577, 0.577]
    ])
    
    G_amps = [0.08, 0.15, 0.30] # T/m
    Delta = 0.030 # s
    delta = 0.012 # s
    
    signals = []
    bvals = []
    
    from dmipy_jax.simulation.simulator import accumulate_phase, trapezoidal_gradient_waveform
    from dmipy_jax.constants import GYRO_MAGNETIC_RATIO
    
    # Pre-calculate time limits
    T_max_pulse = Delta + delta + 0.001
    # We simulated T=50ms. Pulse fits within.
    
    # Trajectories: (N_particles, N_steps, 3)
    # Time vector for trajectories
    num_steps = sol.ys.shape[1]
    t_sim = jnp.linspace(0, T_max, num_steps) # 0 to 50ms
    
    for G in G_amps:
        for d in directions:
            # Generate Waveform G(t)
            G_t = trapezoidal_gradient_waveform(t_sim, G, jnp.array(d), delta, Delta)
            
            # Integrate Phase
            # phases: (N_particles,)
            phases = jax.vmap(accumulate_phase, in_axes=(0, None, None))(sol.ys, G_t, dt)
            
            # Signal
            S = jnp.mean(jnp.exp(1j * phases))
            signals.append(jnp.abs(S))
            
            # Approx b-value ( Stejskal-Tanner )
            # b = (gamma * G * delta)^2 * (Delta - delta/3)
            q = GYRO_MAGNETIC_RATIO * G * delta
            b = q**2 * (Delta - delta/3.0)
            bvals.append(b)
            
    signals = jnp.array(signals)
    bvals = np.array(bvals) / 1e6 # s/mm^2 usually, but here SI?
    # Gyro is usually rad/s/T ~ 2.67e8.
    # q ~ 1e8 * 0.3 * 0.01 ~ 3e5 m^-1
    # b ~ 9e10 * 0.03 ~ 2.7e9 s/m^2 = 2700 s/mm^2. Correct.
    
    print(f"Generated {len(signals)} signal measurements.")
    print(f"B-values (s/mm^2): {bvals[:10] * 1e-6} (Wait, check units)")
    
    # Save Data for Benchmarking
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    np.savez("results/connectome_sde_data.npz", 
             signal=signals, 
             bvals=bvals, 
             gradients=np.tile(directions, (len(G_amps), 1)), # Repeating directions
             ground_truth_fa=FA,
             ground_truth_v1=V1,
             seed_locations=y0,
             final_locations=final_pos
    )
    print("Saved SDE data to results/connectome_sde_data.npz")

    # 7. Visualization
    print("Visualizing visualization...")
    
    # Plot midpoint slice of FA and particles projected
    mid_z = FA.shape[2] // 2
    
    plt.figure(figsize=(10, 10))
    plt.imshow(FA[:, :, mid_z].T, origin='lower', cmap='gray', vmin=0, vmax=1)
    
    # Scatter particles within this slice (z +/- 1)
    mask = jnp.abs(final_pos[:, 2] - mid_z) < 2.0
    
    # Initial positions (Green)
    plt.scatter(y0[mask, 0], y0[mask, 1], c='g', s=10, label='Start', alpha=0.6)
    
    # Final positions (Red)
    plt.scatter(final_pos[mask, 0], final_pos[mask, 1], c='r', s=10, label='End', alpha=0.6)
    
    plt.legend()
    plt.title(f"Data-Driven SDE: Flow in High FA Regions\n(ROI Slice z={mid_z})")
    plt.savefig("demo_connectome_sde_result.png")
    print("Saved demo_connectome_sde_result.png")

if __name__ == "__main__":
    main()
