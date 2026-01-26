import jax
import jax.numpy as jnp
import equinox as eqx
import time
import argparse
from dmipy_jax.models.sh_model import SphericalHarmonicsFit

# Enable x64 by default as requested
jax.config.update("jax_enable_x64", True)

def generate_dummy_data(n_voxels, n_dirs):
    """Generate random signal and gradients."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    # Gradients on sphere
    grads = jax.random.normal(k1, (n_dirs, 3))
    grads = grads / jnp.linalg.norm(grads, axis=1, keepdims=True)
    
    # Signal
    signal = jax.random.uniform(k2, (n_voxels, n_dirs))
    return grads, signal

def benchmark_fit(n_voxels=100_000, n_dirs=256, order=8, chunk_size=4096, device_name="gpu"):
    print(f"Benchmarking SH Fit (Order {order}) on {n_voxels} voxels with {n_dirs} directions.")
    print(f"Device: {device_name.upper()}")
    print(f"Chunk Size: {chunk_size}")

    try:
        if device_name == "cpu":
            dev = jax.devices("cpu")[0]
        else:
            dev = jax.devices("gpu")[0]
    except RuntimeError:
        print(f"Device {device_name} not found. Falling back to default.")
        dev = jax.local_devices()[0]
        
    print(f"Using device: {dev}")
    
    # Move data creation to device? Or create on host and move?
    # Usually data is on host initially.
    grads, signal = generate_dummy_data(n_voxels, n_dirs)
    grads = jax.device_put(grads, dev)
    signal = jax.device_put(signal, dev)
    
    # Initialize Model
    # Since gradients are fixed, pinv is computed once.
    model = SphericalHarmonicsFit(grads, order)
    
    # Define chunked fit
    # We use jax.lax.scan? Or Python loop?
    # Python loop allows us to be sure about memory freeing if needed, 
    # but scan is more idiomatic for "process this big array".
    # Let's use scan for performance.
    
    @jax.jit
    def process_all(model, signal):
        # Reshape signal to (n_chunks, chunk_size, n_dirs)
        # Handle padding if needed
        n_v = signal.shape[0]
        n_chunks = (n_v + chunk_size - 1) // chunk_size
        pad = n_chunks * chunk_size - n_v
        
        if pad > 0:
            signal_padded = jnp.pad(signal, ((0, pad), (0, 0)))
        else:
            signal_padded = signal
            
        signal_reshaped = signal_padded.reshape(n_chunks, chunk_size, n_dirs)
        
        def scan_fn(carry, x):
            # x: (chunk_size, n_dirs)
            return carry, jax.vmap(model)(x)
            
        _, results = jax.lax.scan(scan_fn, None, signal_reshaped)
        
        # Flatten results
        results = results.reshape(-1, results.shape[-1])
        if pad > 0:
            results = results[:n_v]
        return results

    # Warmup
    print("Warming up...")
    _ = process_all(model, signal[:chunk_size])
    jax.block_until_ready(_)
    
    # Run Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    coeffs = process_all(model, signal)
    jax.block_until_ready(coeffs)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    throughput = n_voxels / duration
    
    print(f"Time: {duration:.4f} s")
    print(f"Throughput: {throughput:.2f} voxels/s")
    
    if duration < 60:
        print("[SUCCESS] Fit completed in under 60 seconds.")
    else:
        print("[FAILURE] Fit took longer than 60 seconds.")
        
    return throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_voxels", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()
    
    benchmark_fit(n_voxels=args.n_voxels, device_name=args.device)
