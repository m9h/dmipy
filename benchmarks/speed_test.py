import time
import numpy as np
import jax
import jax.numpy as jnp
# We use mock functions here to test pure throughput without needing the full dmipy-jax codebase loaded yet
from jax import jit

def generate_synthetic_data(n_voxels=100000, n_shells=3, n_dirs=64):
    print(f"Generating data for {n_voxels} voxels...")
    bvals = np.hstack([np.ones(n_dirs) * b for b in [1000, 2000, 3000]])
    bvecs = np.random.randn(len(bvals), 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    data = np.abs(np.random.randn(n_voxels, len(bvals)) + 1j * np.random.randn(n_voxels, len(bvals)))
    return data, bvals, bvecs

def benchmark_jax(data, bvals, bvecs):
    print("--- Starting dmipy-jax (GPU) ---")
    
    # Unified Memory transfer
    data_gpu = jnp.array(data)
    bvals_gpu = jnp.array(bvals)
    bvecs_gpu = jnp.array(bvecs)
    
    # Define a complex mock NODDI kernel
    @jit
    def mock_noddi_fit(d, bval, bvec):
        # Simulation of complex exponential fitting
        E_intra = jnp.exp(-bval[None, :] * 1.7e-3)
        E_extra = jnp.exp(-bval[None, :] * 3.0e-3)
        # Heavy math to stress the tensor cores
        prediction = 0.6 * E_intra + 0.4 * E_extra
        loss = jnp.sum((d - prediction)**2, axis=1)
        return loss

    # Warmup (Compile)
    print("JIT Compiling...")
    mock_noddi_fit(data_gpu[:100], bvals_gpu, bvecs_gpu).block_until_ready()
    
    # The Race
    start = time.time()
    res = mock_noddi_fit(data_gpu, bvals_gpu, bvecs_gpu)
    res.block_until_ready()
    end = time.time()
    
    total_time = end - start
    print(f"⚡ JAX Speed: {len(data)/total_time:.0f} voxels/sec")
    print(f"⏱️ Total Time: {total_time:.4f}s for {len(data)} voxels")

if __name__ == "__main__":
    # Test with 1 Million voxels (Approx 1 whole brain) to flex the GB200
    data, bvals, bvecs = generate_synthetic_data(n_voxels=1_000_000)
    benchmark_jax(data, bvals, bvecs)
