
import jax
import jax.numpy as jnp
import numpy as np
import os
import time
from dmipy_jax.core.invariants import compute_invariants
from dmipy_jax.core.tensor_train import mps_decomposition, extract_angular_core, reconstruct_from_mps

def add_rician_noise(signal, snr):
    """
    Adds Rician noise to the signal.
    Signal S = sqrt( (S_true + n1)^2 + n2^2 )
    where n1, n2 ~ N(0, sigma^2)
    SNR = S0 / sigma -> sigma = S0 / SNR.
    We assume S0 (max signal) is approx 1.0 for normalized signals or we compute it.
    """
    # Estimate S0 from max signal (assuming b=0 is max)
    s0 = jnp.max(signal)
    sigma = s0 / snr
    
    rng = jax.random.PRNGKey(int(time.time()))
    k1, k2 = jax.random.split(rng)
    
    n1 = jax.random.normal(k1, shape=signal.shape) * sigma
    n2 = jax.random.normal(k2, shape=signal.shape) * sigma
    
    noisy_signal = jnp.sqrt((signal + n1)**2 + n2**2)
    return noisy_signal

def main():
    print("Running Noise Robustness Benchmark...")
    
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    try:
        signal = np.load(os.path.join(data_dir, 'phantom_signal.npy'))
        bvecs = np.load(os.path.join(data_dir, 'phantom_bvecs.npy'))
        ground_truth = np.load(os.path.join(data_dir, 'phantom_ground_truth.npy'), allow_pickle=True).item()
    except FileNotFoundError:
        print("Data not found. Please run generate_phantom.py first.")
        return

    # Signal Shape: (Nx, Ny, Nz, N_dirs)
    print(f"Data Shape: {signal.shape}")
    
    # 2. Ground Truth Invariants (from Noiseless Signal)
    print("Computing Ground Truth Invariants...")
    # Reshape to list of voxels for vmap
    # signal_flat = signal.reshape(-1, signal.shape[-1])
    # invariants_gt = compute_invariants(signal_flat, bvecs)
    # This might be too huge for GPU memory if flat? 
    # Vmap `compute_invariants` over x, y, z.
    
    compute_inv_jit = jax.jit(compute_invariants, static_argnames=('max_order',))
    
    # Use simple loop or batching to avoid OOM
    # For 50x50x5 = 12500, it's fine.
    
    # For Invariants:
    # Requires shape (..., N_dirs). compute_invariants supports broadcasting?
    # Yes, my implementation does `dot(..., B_pinv.T)`.
    
    invariants_gt = compute_inv_jit(signal, bvecs, max_order=6)
    
    # 3. Add Noise
    SNR = 20
    print(f"Adding Rician Noise (SNR={SNR})...")
    signal_noisy = add_rician_noise(signal, SNR)
    
    # 4. Method 1: Voxel-wise Fitting
    print("Method 1: Voxel-wise Invariants...")
    t0 = time.time()
    invariants_vox = compute_inv_jit(signal_noisy, bvecs, max_order=6)
    t1 = time.time()
    print(f"Voxel-wise time: {t1-t0:.2f}s")
    
    # 5. Method 2: Tensor Train
    print("Method 2: Tensor Train Decomposition...")
    t0 = time.time()
    
    # Need to reshape signal to 4D for TT: (Nx, Ny, Nz, N_dirs) -> MPS
    # My mps_decomposition takes (N1, N2, N3, N4)
    # Let's use it directly.
    # Ranks?
    # We need to choose ranks that compress but preserve info.
    # Prompt implies TT provides "reduction in parameter variance".
    # This comes from low-rank approximation filtering noise.
    # Let's pick ranks [10, 10, 10]? Or adaptive?
    # For dimensions 50, 50, 5, 64.
    # Ranks: r1 (b/w x,y), r2 (b/w y,z), r3 (b/w z,ang).
    ranks = [20, 20, 20] # Heuristic
    
    cores = mps_decomposition(signal_noisy, ranks)
    
    # Extract Angular Core G4
    # G4: (r3, N_ang, 1) -> squeeze -> (r3, N_ang)
    G4 = extract_angular_core(cores)
    
    # Compute Invariants on Angular Core
    # G4 acts as a set of "basis signals".
    inv_core = compute_inv_jit(G4, bvecs, max_order=6) # Shape (r3, N_inv)
    
    # Reconstruct Full Invariant Map
    # Map back: MPS(G1, G2, G3, Inv_Core)
    # Inv_Core is (r3, N_inv). We need it to be (r3, N_inv, 1) for reconstruct_from_mps?
    # reconstruct_from_mps expects G4: (r3, N4, 1).
    # Here N4 = N_inv.
    
    inv_core_G4 = inv_core[..., None]
    
    invariants_tt = reconstruct_from_mps([cores[0], cores[1], cores[2], inv_core_G4])
    # Shape: (Nx, Ny, Nz, N_inv)
    
    t1 = time.time()
    print(f"TT time: {t1-t0:.2f}s")
    
    # 6. Comparison
    # MSE vs Ground Truth
    mse_vox = jnp.mean((invariants_vox - invariants_gt)**2)
    mse_tt = jnp.mean((invariants_tt - invariants_gt)**2)
    
    print(f"MSE Voxel-wise: {mse_vox:.2e}")
    print(f"MSE Tensor-Train: {mse_tt:.2e}")
    
    improvement = (mse_vox - mse_tt) / mse_vox * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Save results
    results = {
        'mse_vox': mse_vox,
        'mse_tt': mse_tt,
        'improvement': improvement,
        'invariants_gt': invariants_gt,
        'invariants_vox': invariants_vox,
        'invariants_tt': invariants_tt
    }
    np.save(os.path.join(data_dir, 'benchmark_results.npy'), results)

if __name__ == "__main__":
    main()
