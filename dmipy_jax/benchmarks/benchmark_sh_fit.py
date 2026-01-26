import time
import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import os
from dmipy_jax.utils.spherical_harmonics import fit_spherical_harmonics
from dmipy_jax.acquisition import JaxAcquisition

def run_benchmark():
    """
    Benchmark Spherical Harmonics fit on STE00_ExVivo dataset.
    Target: < 60s for the whole volume.
    """
    
    # Path setup
    base_path = os.path.expanduser("~/Downloads/STE/STE00_ExVivo/STE")
    nii_path = os.path.join(base_path, "STE_degibbs_eddy.nii.gz")
    bval_path = os.path.join(base_path, "bvals.txt")
    bvec_path = os.path.join(base_path, "bvecs.txt")
    
    if not os.path.exists(nii_path):
        print(f"Dataset not found at {nii_path}. benchmarking with random data.")
        # Fallback to random data if user didn't download or path wrong
        # But per task we must use this data.
        # Let's fail if not found to ensure we test the right thing per user request.
        raise FileNotFoundError(f"Required benchmark dataset not found: {nii_path}")

    print("Loading data...")
    t0 = time.time()
    img = nib.load(nii_path)
    data = img.get_fdata() # (X, Y, Z, N_dirs)
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path).T # (N_dirs, 3)
    
    # Fix potential bvec shape issue if read as (3, N)
    if bvecs.shape[1] != 3 and bvecs.shape[0] == 3:
        bvecs = bvecs.T

    # Extract shell b=1000 (approx)
    # The file has b=0 and b=1000.
    # We want to benchmark the fit on the diffusion weighted images.
    # But usually SH fit is done on normalized signal or raw signal?
    # Usually we fit to a specific shell.
    # Let's take the b~1000 shell.
    
    mask_b1000 = (bvals > 900) & (bvals < 1100)
    data_shell = data[..., mask_b1000]
    bvecs_shell = bvecs[mask_b1000]
    bvals_shell = bvals[mask_b1000]
    
    print(f"Data shape: {data.shape}")
    print(f"Shell extraction (b=1000): {data_shell.shape} voxels, {bvecs_shell.shape[0]} directions.")
    print(f"Load time: {time.time() - t0:.2f}s")
    
    # Prepare JAX objects
    # Normalize signal by b0?
    # Simple SH fit often assumes S/S0. 
    # Let's compute S0 (b < 50)
    mask_b0 = bvals < 50
    s0 = np.mean(data[..., mask_b0], axis=-1)
    # Avoid div by zero
    s0[s0 == 0] = 1.0
    
    # Normalize
    # We need to broadcast S0: (X, Y, Z, 1)
    signal_normalized = data_shell / s0[..., None]
    
    # Move to JAX
    signal_jax = jnp.array(signal_normalized)
    
    acq = JaxAcquisition(
        bvalues=jnp.array(bvals_shell),
        gradient_directions=jnp.array(bvecs_shell)
    )
    
    # Warmup / Compilation
    print("\nTriggering JIT compilation (on single voxel)...")
    # Take middle voxel
    mid = [s // 2 for s in signal_jax.shape[:3]]
    voxel = signal_jax[mid[0], mid[1], mid[2]]
    
    # Compile the fit function specialized for this acquisition shape
    fit_compiled = jax.jit(lambda s: fit_spherical_harmonics(s, acq, lmax=8))
    _ = fit_compiled(voxel) # Run once
    print("Compilation done.")
    
    # Full Volume Benchmark
    print("\nRunning Full Volume Benchmark (GPU if available)...")
    
    # We can rely on fit_spherical_harmonics handling the batching via dot product broadcasting
    # or we can explicitly vmap if needed. 
    # The implementation uses simple dot product (batch_dims @ matrix), which is efficient.
    # fit_spherical_harmonics(signal, acq)
    # signal: (X, Y, Z, N_grads)
    
    # We should ensure the computation happens on device.
    # We'll block until ready.
    
    t_start = time.time()
    coeffs = fit_compiled(signal_jax)
    coeffs.block_until_ready()
    t_end = time.time()
    
    duration = t_end - t_start
    print(f"\nBenchmark Result:")
    print(f"Total time: {duration:.4f} s")
    print(f"Voxel count: {np.prod(signal_jax.shape[:3])}")
    print(f"Directions: {bvecs_shell.shape[0]}")
    print(f"Coefficients shape: {coeffs.shape}")
    
    if duration < 60.0:
        print("SUCCESS: Performance target < 60s met.")
    else:
        print("FAILURE: Performance target < 60s NOT met.")
        exit(1)

if __name__ == "__main__":
    run_benchmark()
