import sys
import jax
import jax.numpy as jnp
import os

print("V2 Start", flush=True)

# Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from dmipy_jax.examples.sbi.train_global import GlobalMDN, get_global_batch, loss_fn, sample_random_acquisition
    from dmipy_jax.inference.tracker import ProbabilisticTracker
    from dmipy_jax.signal_models import gaussian_models
    print("Imports Done", flush=True)
except Exception as e:
    print(f"Import Failed: {e}", flush=True)
    sys.exit(1)

# Init Key
try:
    key = jax.random.PRNGKey(0)
    print(f"Key Init: {key}", flush=True)
except Exception as e:
    print(f"Key Failed: {e}", flush=True)
    sys.exit(1)

# Init Model
try:
    print("Initializing Model...", flush=True)
    max_n = 32
    model = GlobalMDN(key, max_n=max_n, n_outputs=6)
    print("Model Initialized", flush=True)
except Exception as e:
    print(f"Model Init Failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Mock Data
try:
    print("Getting Batch...", flush=True)
    x, y = get_global_batch(key, batch_size=16, max_n_dirs=max_n)
    print(f"Batch Got: {x.shape}", flush=True)
except Exception as e:
    print(f"Batch Failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Inference
try:
    print("Running Inference...", flush=True)
    loss = loss_fn(model, x, y)
    print(f"Loss: {loss}", flush=True)
except Exception as e:
    print(f"Inference Failed: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
    
print("V2 Success", flush=True)
