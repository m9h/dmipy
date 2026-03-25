
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dmipy_jax.io.wand import WANDLoader
from dmipy_jax.denoise.patch2self import Patch2Self

def main():
    print("Initializing Patch2Self Demo...")
    loader = WANDLoader()
    
    # Define ROI - Large enough for statistics, small enough for quick demo
    # WAND data is usually (128, 128, 60, N_meas)
    # Let's take a central block
    roi = (slice(30, 90), slice(30, 90), slice(20, 26)) 
    
    try:
        print("Loading WAND Data (ROI)...")
        data_dict = loader.load_axcaliber_data(roi_slice=roi)
    except Exception as e:
        print(f"Initial load failed: {e}")
        print("Attempting to fetch data...")
        try:
            loader.fetch_data()
            data_dict = loader.load_axcaliber_data(roi_slice=roi)
        except Exception as e2:
            print(f"Could not load WAND data: {e2}")
            print("Falling back to SYNTHETIC data generation.")
            # Generate Synthetic Data
            # (64, 64, 64, 32)
            dims = (64, 64, 20, 32)
            print(f"Generating synthetic phantom {dims}...")
            
            # Create a simple structure: Sphere in center
            x = jnp.linspace(-1, 1, dims[0])
            y = jnp.linspace(-1, 1, dims[1])
            z = jnp.linspace(-1, 1, dims[2])
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            mask = (X**2 + Y**2 + Z**2) < 0.6
            
            # Signal: Decay based on b-value?
            # Random b-vals
            bvals = jnp.linspace(0, 3000, dims[3])
            
            # Signal = S0 * exp(-b * D)
            D = 1e-3 # diffusivity
            S0 = 100.0
            signal_decay = S0 * jnp.exp(-bvals[None, None, None, :] * D)
            
            # Modulate with mask (Background = 0)
            ground_truth = signal_decay * mask[..., None]
            
            # Add Noise (Rician = Sqrt((S + N1)^2 + N2^2))
            key = jax.random.PRNGKey(42)
            k1, k2 = jax.random.split(key)
            noise_std = 10.0
            n1 = noise_std * jax.random.normal(k1, ground_truth.shape)
            n2 = noise_std * jax.random.normal(k2, ground_truth.shape)
            
            data = jnp.sqrt((ground_truth + n1)**2 + n2**2)
            
            data_dict = {
                'data': data,
                'bvals': bvals
            }
            print(f"Synthetic Data SNR: {S0/noise_std:.1f}")

    data = data_dict['data'] # (X, Y, Z, B)
    bvals = data_dict['bvals']
    print(f"Data Shape: {data.shape}")
    print(f"B-values: {bvals.shape}")

    # Initialize Patch2Self
    # Radius 2 -> 5x5x5 patches = 125 voxels per volume context
    p2s = Patch2Self(patch_radius=[2, 2, 0], model='ols') 
    # Note: radius [2,2,0] is 2D patches in 3D volume? 
    # Or strict 3D: [1,1,1] -> 3x3x3.
    # WAND has anisotropic resolution? Usually 2mm isotropic?
    # Let's use standard radius=1 (3x3x3).
    p2s = Patch2Self(patch_radius=1, model='ols')

    print("Running Patch2Self Denoising...")
    start_time = time.time()
    
    # JIT compile the first run
    denoised_data = p2s(data, bvals)
    # Block until ready
    denoised_data.block_until_ready()
    
    duration = time.time() - start_time
    print(f"Denoising completed in {duration:.2f} seconds.")
    print(f"Throughput: {data.size / duration / 1e6:.2f} MVoxels/s")

    # Visualization
    # Pick a high b-value shell
    high_b_idx = jnp.argmax(bvals)
    b_max = bvals[high_b_idx]
    
    print(f"Visualizing results for b={b_max} (Index {high_b_idx})")
    
    z_slice = data.shape[2] // 2
    
    raw_img = data[:, :, z_slice, high_b_idx]
    den_img = denoised_data[:, :, z_slice, high_b_idx]
    residual = raw_img - den_img
    
    # Calculate RMSE
    rmse = jnp.sqrt(jnp.mean(residual**2))
    print(f"RMSE: {rmse:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(raw_img.T, origin='lower', cmap='gray', vmin=0, vmax=np.percentile(raw_img, 99))
    plt.title(f"Raw (b={b_max:.0f})")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(den_img.T, origin='lower', cmap='gray', vmin=0, vmax=np.percentile(raw_img, 99))
    plt.title("Patch2Self Denoised")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(residual.T, origin='lower', cmap='gray')
    plt.title(f"Residuals (RMSE={rmse:.2f})")
    plt.colorbar()
    
    out_file = "dmipy_jax/examples/denoising/patch2self_demo.png"
    # Ensure dir exists
    import os
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
