# experiments/ste_dataset_integration/benchmark_sh.py
import time
import jax
import jax.numpy as jnp
import numpy as np

# Enable x64 for precision as per protocol
jax.config.update("jax_enable_x64", True)

def get_dummy_data(batch_size=4096, n_dirs=256):
    """Generates dummy signal and gradient directions."""
    print(f"Generating dummy data: {batch_size} voxels, {n_dirs} directions...")
    key = jax.random.PRNGKey(0)
    # Random signal: Batch x N_dirs
    signals = jax.random.uniform(key, (batch_size, n_dirs))
    # Random directions: N_dirs x 3 (normalized)
    grad_key, _ = jax.random.split(key)
    gradients = jax.random.normal(grad_key, (n_dirs, 3))
    gradients = gradients / jnp.linalg.norm(gradients, axis=1, keepdims=True)
    return signals, gradients

def benchmark_sh_fit():
    BATCH_SIZE = 4096 * 4 # Increase load for clearer GPU stats
    N_DIRS = 256
    ORDER = 8
    
    print(f"Benchmarking SH Fit (Order={ORDER})")
    print(f"Batch Size: {BATCH_SIZE}, Directions: {N_DIRS}")
    print(f"Device: {jax.devices()[0]}")

    signals, gradients = get_dummy_data(BATCH_SIZE, N_DIRS)
    
    # Create a random design matrix to simulate SH basis
    # In a real app, this depends on gradients.
    n_coeffs = int((ORDER + 1) * (ORDER + 2) / 2)
    key = jax.random.PRNGKey(1)
    design_matrix = jax.random.normal(key, (N_DIRS, n_coeffs))
    
    print(f"Design Matrix Shape: {design_matrix.shape}")
    
    # We want to map over voxels.
    # Signal shape: (Batch, N_dirs)
    # Design matrix: (N_dirs, N_coeffs)
    # Formula: beta = (X^T X)^-1 X^T y
    # jnp.linalg.lstsq solves x for Ax=b. Here X*beta = signal^T ? 
    # Usually S = X * beta. So fit is beta = lstsq(X, S).
    # Since S is (N_dirs,), we can effectively do this.
    
    @jax.jit
    def fit_voxel(signal):
        # returns coeffs, residual, rank, s
        return jnp.linalg.lstsq(design_matrix, signal, rcond=None)[0]
    
    # Batch fit
    batch_fit = jax.jit(jax.vmap(fit_voxel))
    
    # Warmup
    print("Warming up JIT...")
    _ = batch_fit(signals)
    _.block_until_ready()
    
    # Benchmark
    print("Running benchmark...")
    t0 = time.perf_counter()
    N_LOOPS = 20
    for _ in range(N_LOOPS):
        coeffs = batch_fit(signals)
        coeffs.block_until_ready()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    total_voxels = BATCH_SIZE * N_LOOPS
    voxels_per_sec = total_voxels / total_time
    
    print(f"Total Time: {total_time:.4f} s")
    print(f"Processed: {total_voxels} voxels")
    print(f"Throughput: {voxels_per_sec:,.2f} voxels/sec")
    
    if voxels_per_sec > 100000:
        print("PERFORMANCE: EXCELLENT (>100k)")
    else:
        print("PERFORMANCE: NORMAL")

if __name__ == "__main__":
    benchmark_sh_fit()
