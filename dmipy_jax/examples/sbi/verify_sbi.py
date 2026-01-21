import jax
import jax.numpy as jnp
import equinox as eqx
import sys
import os

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.examples.sbi.train_global import GlobalMDN, get_global_batch, loss_fn, sample_random_acquisition
from dmipy_jax.inference.tracker import ProbabilisticTracker
from dmipy_jax.signal_models import gaussian_models

def main():
    print("--- VERIFICATION: Global SBI & Tracking ---", flush=True)
    
    try:
        # 1. Initialize Model
        key = jax.random.PRNGKey(0)
        max_n = 32
        print("Initializing Model...", flush=True)
        model = GlobalMDN(key, max_n=max_n, n_outputs=6)
        
        # 2. Test Training Logic
        print("Testing Batch Generation...", flush=True)
        x, y = get_global_batch(key, batch_size=16, max_n_dirs=max_n)
        print(f"Input: {x.shape}, Target: {y.shape}", flush=True)
        
        loss = loss_fn(model, x, y)
        print(f"Loss: {loss}", flush=True)
        
        # 3. Test Tracker
        print("Testing Probabilistic Tracker...", flush=True)
        k_trk = jax.random.PRNGKey(1)
        
        # Create minimal acquisition (6 dirs)
        # We need bvals, bvecs to initialize tracker
        bvals, bvecs, mask = sample_random_acquisition(k_trk, max_n=max_n, min_n=6)
        
        # Filter valid ones for Tracker init
        valid_idx = jnp.where(mask)[0]
        # Ensure we have at least some? min_n=6
        # Take first 6 valid
        bvals_meas = bvals[valid_idx][:6]
        bvecs_meas = bvecs[valid_idx][:6]
        
        print(f"Tracking with {len(bvals_meas)} directions.", flush=True)
        
        # Setup Tracker
        tracker = ProbabilisticTracker(model, bvals_meas, bvecs_meas, step_size=0.5, max_dirs=max_n)
        
        # Create dummy volume (single voxel repeated)
        # Signal of 6 measurements
        sig_meas = jnp.ones(len(bvals_meas)) * 0.5 # Dummy signal
        
        dim = 5
        vol = jnp.zeros((dim, dim, dim, len(bvals_meas)))
        vol = vol.at[2, 2, 2].set(sig_meas)
        
        seed = jnp.array([2.5, 2.5, 2.5])
        
        # Run Track
        print("Tracing Streamline...", flush=True)
        streamline, valid = tracker.track(seed, vol, None, k_trk, max_steps=10)
        
        print(f"Streamline Points: {jnp.sum(valid)}", flush=True)
        print(f"Start: {streamline[0]}", flush=True)
        print(f"End: {streamline[1]}", flush=True)
        
        print("Verification Successful.", flush=True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL FAILURE: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
