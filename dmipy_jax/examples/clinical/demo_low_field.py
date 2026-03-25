
import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.io.lowfield import load_ds006557_data

def run_low_field_demo():
    print("=== Low Field MRI (64mT) Denoising Demo ===")
    print("Demonstrating JAX-accelerated Total Variation (TV) Denoising on OpenNeuro ds006557.")

    # 1. Load Data
    print("\n--> Fetching Data (sub-HYPE00)...")
    try:
        # T1-weighted image from Hyperfine scanner
        data_dict = load_ds006557_data(subject="sub-HYPE00", contrast="T1w")
        img_data = data_dict['image']
        print(f"Loaded Image Shape: {img_data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Extract a slice for visualization (and speed demo)
    # Middle slice
    nz = img_data.shape[2]
    slice_idx = nz // 2
    img_slice = img_data[:, :, slice_idx]
    
    # Normalize to [0, 1]
    img_slice = img_slice / jnp.max(img_slice)
    
    print(f"Selected Slice {slice_idx}. Intensity Range: [{jnp.min(img_slice):.2f}, {jnp.max(img_slice):.2f}]")

    # 2. Define TV Denoising Solver (ROF Model)
    # Minimize: ||x - y||^2 + lambda * TV(x)
    # Using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for TV usually involves dual approach
    # Or simple Projected Gradient Descent on the Dual.
    # Here we use a simple Gradient Descent on the Primal with smoothed TV (Huber) for simplicity and JAX autodiff.
    
    lambda_tv = 0.05 # Strength of regularization
    huber_delta = 1e-3 # Smoothing parameter
    
    @jax.jit
    def tv_huber(x):
        # x: (H, W)
        diff_x = x[1:, :] - x[:-1, :]
        diff_y = x[:, 1:] - x[:, :-1]
        
        # L1 approx via Huber: sqrt(x^2 + delta^2) - delta
        tv_x = jnp.sum(jnp.sqrt(diff_x**2 + huber_delta**2) - huber_delta)
        tv_y = jnp.sum(jnp.sqrt(diff_y**2 + huber_delta**2) - huber_delta)
        return tv_x + tv_y

    @jax.jit
    def loss_fn(x, y_noisy):
        mse = 0.5 * jnp.sum((x - y_noisy)**2)
        reg = lambda_tv * tv_huber(x)
        return mse + reg

    @jax.jit
    def update_step(x, lr, y_noisy):
        grads = jax.grad(loss_fn)(x, y_noisy)
        x_new = x - lr * grads
        # Projection to valid range [0, 1] (Physical constraint)
        x_new = jnp.clip(x_new, 0.0, 1.0)
        return x_new

    # 3. Process
    print("\n--> Running JAX-TV Optimization (Huber-TV)...")
    
    # Init with noisy image
    x_est = img_slice
    lr = 0.05 # Learning rate
    n_iter = 200
    
    start_time = time.time()
    for i in range(n_iter):
        x_est = update_step(x_est, lr, img_slice)
        if i % 50 == 0:
            l_val = loss_fn(x_est, img_slice)
            print(f"Iter {i}: Loss = {l_val:.5f}")
            
    jax.block_until_ready(x_est)
    duration = time.time() - start_time
    print(f"Optimization finished in {duration:.2f}s ({duration/n_iter*1000:.2f} ms/iter)")

    # 4. Visualization & Metrics
    print("\n--> Generating Report...")
    
    # Calculate noise estimate (sigma) in background
    # Simple heuristic: corner region
    bg_patch = img_slice[:20, :20]
    bg_std = jnp.std(bg_patch)
    
    bg_patch_den = x_est[:20, :20]
    bg_std_den = jnp.std(bg_patch_den)
    
    print(f"Background Noise (StdDev): Original={bg_std:.4f} -> Denoised={bg_std_den:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.rot90(img_slice), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original Low-Field (64mT)")
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(x_est), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"JAX TV Denoised (lambda={lambda_tv})")
    axes[1].axis('off')
    
    # Difference
    diff = jnp.abs(img_slice - x_est)
    im_diff = axes[2].imshow(np.rot90(diff), cmap='inferno')
    axes[2].set_title("Residuals (Noise Removed)")
    axes[2].axis('off')
    plt.colorbar(im_diff, ax=axes[2])
    
    out_file = "low_field_demonstration.png"
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")
    
    print("\nSUCCESS: Low Field Demonstration Completed.")

if __name__ == "__main__":
    run_low_field_demo()
