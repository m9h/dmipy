import jax
import jax.numpy as jnp
import numpy as np
import os
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin
from dmipy_jax.sphere import S2SphereStejskalTannerApproximation as S2Sphere
from dmipy_jax.acquisition import JaxAcquisition

def generate_hcp_b_table(shells, directions_per_shell, b0_count=20):
    """
    Generates a synthetic HCP-like b-table.
    Copied from synthetic_data.py to avoid import issues.
    """
    bvals = []
    bvecs = []
    
    # Add b0s
    for _ in range(b0_count):
        bvals.append(0.0)
        bvecs.append([0.0, 0.0, 0.0])
        
    for shell_b, n_dirs in zip(shells, directions_per_shell):
        indices = np.arange(0, n_dirs, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_dirs)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        vecs = np.stack([x, y, z], axis=1)
        
        for v in vecs:
            bvals.append(shell_b)
            v = v / np.linalg.norm(v)
            bvecs.append(v)
            
    return np.array(bvals), np.array(bvecs)

def get_hcp_mgh_scheme():
    # HCP MGH b-table
    shells = [1000, 3000, 5000, 10000]
    dirs = [64, 64, 128, 256]
    bvals_si, bvecs = generate_hcp_b_table(shells, dirs)
    
    # Convert to SI units (s/m^2)
    bvals_si = bvals_si * 1e6 
    
    # Mock delta/Delta for acquisition
    # HCP-MGH: Delta=21.8ms, delta=12.9ms (approx)
    Delta = 21.8e-3
    delta = 12.9e-3
    
    scheme = JaxAcquisition(
        bvalues=jnp.array(bvals_si),
        gradient_directions=jnp.array(bvecs),
        delta=jnp.full(len(bvals_si), delta),
        Delta=jnp.full(len(bvals_si), Delta)
    )
    return scheme, bvals_si, bvecs

def simulate_group(scheme, n_voxels, f_stick, f_sphere, snr=30):
    # Parameters
    # Fiber direction: X-axis [pi/2, 0]
    mu = jnp.array([jnp.pi/2, 0.0])
    
    # Diffusivities
    lambda_par = 1.7e-9
    lambda_perp = 0.3e-9 # Typical for Zeppelin
    diameter = 10e-6 # 10 microns for soma
    
    # Fractions
    f_zeppelin = 1.0 - f_stick - f_sphere
    
    if f_zeppelin < 0:
        raise ValueError("Sum of fractions > 1.0")

    # Instantiate Models
    stick = C1Stick(mu=mu, lambda_par=lambda_par)
    zeppelin = G2Zeppelin(mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp)
    sphere = S2Sphere(diameter=diameter)
    
    # Simulate (Noise Free) using vmap if needed, but params are constant here.
    # We iterate 50 times for noise generation later, or generate 50 identical signals.
    
    E_stick = stick(bvals=scheme.bvalues, gradient_directions=scheme.gradient_directions)
    E_zeppelin = zeppelin(bvals=scheme.bvalues, gradient_directions=scheme.gradient_directions)
    E_sphere = sphere(qvalues=scheme.qvalues) 
    
    # Weighted Sum
    S_clean = f_stick * E_stick + f_zeppelin * E_zeppelin + f_sphere * E_sphere
    
    # Replicate for n_voxels
    S_batch = jnp.tile(S_clean, (n_voxels, 1))
    
    # Add Rician Noise
    # Noise sigma = 1/SNR (assuming S0=1)
    sigma = 1.0 / snr
    
    # Use numpy for noise generation
    S_batch_np = np.array(S_batch)
    
    # Basic Rician noise: sqrt((S+n1)^2 + n2^2)
    n1 = np.random.normal(0, sigma, S_batch_np.shape)
    n2 = np.random.normal(0, sigma, S_batch_np.shape)
    
    S_noisy = np.sqrt((S_batch_np + n1)**2 + n2**2)
    
    return S_noisy

def main():
    print("Generating Delta-Phantom...")
    
    scheme, bvals, bvecs = get_hcp_mgh_scheme()
    
    # Group A
    print("Simulating Group A (Baseline, f_sphere=0.1)...")
    signal_A = simulate_group(scheme, n_voxels=50, f_stick=0.5, f_sphere=0.1, snr=30)
    np.save('group_A_signal.npy', signal_A)
    
    # Group B
    print("Simulating Group B (Lesion, f_sphere=0.2)...")
    signal_B = simulate_group(scheme, n_voxels=50, f_stick=0.5, f_sphere=0.2, snr=30)
    np.save('group_B_signal.npy', signal_B)
    
    # Verification
    # Calculate Mean SDNR
    # Definition of SDNR here: |Mean(S_A) - Mean(S_B)| / sigma
    # This represents the visibility of the difference given the noise level.
    
    mean_A = np.mean(signal_A, axis=0) # Mean signal A vector
    mean_B = np.mean(signal_B, axis=0) # Mean signal B vector
    
    diff = np.abs(mean_A - mean_B)
    
    # Noise standard deviation (approximate from SNR=30, sigma=1/30)
    sigma = 1.0 / 30.0
    
    # SDNR per measurement
    sdnr_per_meas = diff / sigma
    
    mean_sdnr = np.mean(sdnr_per_meas)
    
    print(f"Mean Signal Difference-to-Noise Ratio (SDNR): {mean_sdnr:.4f}")
    
    if mean_sdnr < 1.0:
        print("WARNING: SDNR < 1.0. Change might be too subtle to detect.")
    
    # Also save bvals/bvecs for reference
    np.savetxt('delta_phantom.bval', bvals[None, :], fmt='%d')
    np.savetxt('delta_phantom.bvec', bvecs.T, fmt='%.6f')
    
    print("Done! Saved group_A_signal.npy, group_B_signal.npy, delta_phantom.bval, delta_phantom.bvec")

if __name__ == "__main__":
    main()
