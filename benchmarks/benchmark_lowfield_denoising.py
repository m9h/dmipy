
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dmipy_jax.io.lowfield import load_ds006557_data
from dmipy_jax.io.mne_data import load_mne_sample_data

# Algorithms
def denoise_gaussian(image, sigma=1.0):
    """Simple Gaussian smoothing as a baseline."""
    # Simple convolution implementation or use jax.scipy
    from jax.scipy.ndimage import map_coordinates
    # We can use a simple kernel for speed benchmark
    # But let's use something that is actually doing 'denoising'
    # For benchmark, we might just use a placeholder if we don't have a library.
    # But wait, we should check if we can use scico or dmipy_jax's own tools.
    # If not, let's implement a simple JAX TV denoising.
    return image # Placeholder if not implemented inline, but see below.

def total_variation_denoising(image, weight=0.1, n_iter=50):
    """
    Simple JAX implementation of Total Variation Denoising (ROF model)
    using Primal-Dual Hybrid Gradient (Chambolle-Pock).
    """
    # x: original, u: denoised
    # min ||u - x||^2 + lambda * ||grad u||
    
    u = image
    px = jnp.zeros_like(image)
    py = jnp.zeros_like(image)
    pz = jnp.zeros_like(image)
    
    tau = 0.125
    sigma = 0.125 # preconditioners? simple constant step for now
    
    def step(carry, _):
        u, px, py, pz = carry
        
        # Dual update (divergence)
        # div p = dx px + dy py + dz pz
        # Finite differences
        div_p = (jnp.roll(px, 1, axis=0) - px) + \
                (jnp.roll(py, 1, axis=1) - py) + \
                (jnp.roll(pz, 1, axis=2) - pz)
                
        # Primal update
        u_prev = u
        u = (u + tau * div_p + tau * image) / (1 + tau) # Prox for L2 data fidelity
        
        # Extrapolation
        u_bar = 2 * u - u_prev
        
        # Dual update (gradient)
        grad_x = jnp.roll(u_bar, -1, axis=0) - u_bar
        grad_y = jnp.roll(u_bar, -1, axis=1) - u_bar
        grad_z = jnp.roll(u_bar, -1, axis=2) - u_bar
        
        px = (px + sigma * grad_x)
        py = (py + sigma * grad_y)
        pz = (pz + sigma * grad_z)
        
        # Projection (L1 ball for TV)
        norm = jnp.sqrt(px**2 + py**2 + pz**2)
        scale = jnp.maximum(1.0, norm / weight)
        
        px /= scale
        py /= scale
        pz /= scale
        
        return (u, px, py, pz), None

    (u_final, _, _, _), _ = jax.lax.scan(step, (u, px, py, pz), None, length=n_iter)
    return u_final

# JIT compile with static n_iter (index 2)
tv_denoise_jit = jax.jit(total_variation_denoising, static_argnums=(2,))


def benchmark_denoising():
    print("=== Low Field vs MNE Denoising Benchmark ===")
    
    results = []
    
    datasets = []
    
    # 1. Load MNE Data
    try:
        mne_data = load_mne_sample_data()
        datasets.append(("MNE Sample", mne_data['image']))
    except Exception as e:
        print(f"Skipping MNE: {e}")
        
    # 2. Load Low Field Data
    try:
        # Use sub-01 as standard BIDS
        lf_data = load_ds006557_data(subject="sub-01")
        datasets.append(("OpenNeuro LowField", lf_data['image']))
    except Exception as e:
        print(f"Skipping LowField: {e}")
        
    if not datasets:
        print("No datasets loaded.")
        return

    # Benchmark Loop
    for name, img in datasets:
        print(f"\nBenchmarking {name} (Shape: {img.shape})")
        
        # Normalize roughly to 0-1 for stability
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        # Crop or resize if too huge? 
        # For MNE T1 (256^3), it's manageable on CPU/GPU.
        
        # Metrics to track
        n_voxels = img.size
        
        # Warmup
        print("  Warming up...")
        _ = tv_denoise_jit(img_norm, n_iter=5).block_until_ready()
        
        # Run
        print("  Running TV Denoising (50 iters)...")
        t0 = time.time()
        denoised = tv_denoise_jit(img_norm, n_iter=50).block_until_ready()
        duration = time.time() - t0
        
        print(f"  Done in {duration:.4f}s")
        print(f"  Throughput: {n_voxels/duration/1e6:.2f} MVoxels/s")
        
        # Simple noise estimate (sigma)
        # Assume corner background
        bg_slice = img_norm[:20, :20, :20]
        sigma_est = jnp.std(bg_slice)
        print(f"  Est. input noise sigma: {sigma_est:.4f}")
        
        results.append({
            "Dataset": name,
            "Voxels": n_voxels,
            "Time (s)": duration,
            "Throughput (MVox/s)": n_voxels/duration/1e6,
            "Noise Est": float(sigma_est)
        })
        
    # Report
    df = pd.DataFrame(results)
    print("\n--- Results ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    benchmark_denoising()
