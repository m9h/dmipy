
import time
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import nibabel as nib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dmipy_jax.io.multi_te import MultiTELoader
from dmipy_jax.denoise.mppca import mppca

def benchmark():
    print("Initializing Benchmark...")
    
    # Path to data
    # Assuming standard location or accessible
    # Use the same path as in other examples or explicit
    base_path = "/mnt/data/multi_te_2025" # Update if needed, or check environment
    # Fallback to a Check
    if not os.path.exists(base_path):
        # Try local dev path
        base_path = "/home/mhough/dev/dmipy/data/multi_te_2025" # Example
        if not os.path.exists(base_path):
             print(f"Warning: Dataset not found at {base_path}. Using synthetic data.")
             synthetic = True
        else:
             synthetic = False
    else:
        synthetic = False
        
    if not synthetic:
        print(f"Loading data from {base_path}")
        subject = 'sub-03'
        try:
            loader = MultiTELoader(base_path, subject)
            tes = loader.get_available_tes()
            print(f"Available TEs: {tes}")
            target_te = tes[0] # Pick first
            print(f"Loading TE={target_te}...")
            data_jax, bvals, bvecs, prot = loader.load_data(target_te)
            data = np.array(data_jax) # Convert to numpy for dipping/plotting compatibility
            affine = loader.load_image_affine(target_te)
        except Exception as e:
            print(f"Failed to load data: {e}")
            synthetic = True
            
    if synthetic:
        print("Generating synthetic 4D data...")
        shape = (60, 60, 30, 64)
        data = np.random.randn(*shape).astype(np.float32) + 10.0
        # Add structure
        data[:, :, :, :10] += 50.0
        affine = np.eye(4)
        
    print(f"Data Shape: {data.shape}")
    
    # Run JAX MP-PCA
    print("\n--- Running JAX MP-PCA ---")
    
    # Warmup
    print("Warming up (compiling)...")
    start = time.time()
    # Convert to JAX array
    data_j = jnp.array(data)
    _ = mppca(data_j, patch_radius=2).block_until_ready()
    end = time.time()
    print(f"Warmup time: {end - start:.4f} s")
    
    # Benchmark
    print("Benchmarking...")
    start = time.time()
    denoised_jax = mppca(data_j, patch_radius=2).block_until_ready()
    end = time.time()
    jax_time = end - start
    print(f"JAX Execution time: {jax_time:.4f} s")
    print(f"Throughput: {data.size / jax_time / 1e6:.2f} MVoxels/s")
    
    # Run DIPY Baseline if available
    try:
        from dipy.denoise.localpca import mppca as dipy_mppca
        print("\n--- Running DIPY MP-PCA (CPU) ---")
        start = time.time()
        # dipy mppca takes data and patch_radius
        # Note: DIPY might use 'patch_radius' or 'patch_size'. 
        # Check signature: mppca(arr, mask=None, patch_radius=2, pca_method='svd', return_sigma=False, out_dtype=None)
        denoised_dipy = dipy_mppca(data, patch_radius=2)
        end = time.time()
        dipy_time = end - start
        print(f"DIPY Execution time: {dipy_time:.4f} s")
        print(f"Speedup: {dipy_time / jax_time:.2f}x")
        
        # Compare results
        diff = np.abs(denoised_dipy - np.array(denoised_jax))
        mae = np.mean(diff)
        print(f"Mean Absolute Difference (JAX vs DIPY): {mae:.6f}")
        
    except ImportError:
        print("\nDIPY not installed or failed to import. Skipping baseline.")
    except Exception as e:
        print(f"\nDIPY Baseline failed: {e}")
        
    # Save Output
    out_name = f"denoised_mppca_jax.nii.gz"
    print(f"\nSaving JAX result to {out_name}")
    nib.save(nib.Nifti1Image(np.array(denoised_jax), affine), out_name)
    print("Done.")

if __name__ == "__main__":
    benchmark()
