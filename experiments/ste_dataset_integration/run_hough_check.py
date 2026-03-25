import jax
import jax.numpy as jnp
import time
from dmipy_jax.tractography.hough_global import global_hough_tracking

def run_check():
    print("--- Verifying Hough Global Module ---")
    
    # 1. Dummy SH Volume (16x16x16, 15 coeffs for L=4)
    vol_shape = (16, 16, 16, 15)
    print(f"Creating dummy SH volume {vol_shape}...")
    sh_vol = jnp.zeros(vol_shape)
    # Set L=0 coeff to 1.0 (isotropic)
    sh_vol = sh_vol.at[..., 0].set(1.0)
    # Add some anisotropy in x-direction (L=2, m=2 term?)
    # Just random noise to make scoring interesting
    key = jax.random.PRNGKey(0)
    sh_vol = sh_vol + 0.1 * jax.random.normal(key, vol_shape)
    
    # 2. Seeds
    n_seeds = 100
    print(f"Creating {n_seeds} seeds...")
    seeds = jnp.array([[8.0, 8.0, 8.0]] * n_seeds) + jax.random.normal(key, (n_seeds, 3))
    
    # 3. Run Tracking
    print("Running global_hough_tracking (JIT compile)...")
    t0 = time.time()
    
    # JIT the main function
    track_fn = jax.jit(lambda v, s: global_hough_tracking(v, s, n_hough_samples=5000))
    
    curves, scores = track_fn(sh_vol, seeds)
    curves.block_until_ready()
    
    duration = time.time() - t0
    print(f"Done in {duration:.4f}s")
    
    print(f"Output Curves Shape: {curves.shape}")
    print(f"Output Scores Shape: {scores.shape}")
    print(f"Mean Score: {jnp.mean(scores)}")
    
    if jnp.any(jnp.isnan(scores)):
        print("FAIL: NaNs detected in scores.")
    else:
        print("SUCCESS: Scores are valid.")

if __name__ == "__main__":
    run_check()
