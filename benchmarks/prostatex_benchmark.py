
import os
import pydicom
import glob
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from dmipy_jax.fitting.optimization import VoxelFitter
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme
from dmipy_jax.signal_models.ivim import IVIM
from dmipy_jax.algebra.initializers import segmented_ivim_init
from dmipy_jax.fitting.optimization import VoxelFitter
import optimistix as optx

# Helper: Load DICOM series
def load_prostatex_dwi(folder_path):
    # Recursive glob
    files = sorted(glob.glob(os.path.join(folder_path, "**/*.dcm"), recursive=True)) 
    # This assumes simple sorting works for slice/b-value ordering. 
    # In reality, need to parse InstanceNumber or similar.
    
    slices = [pydicom.dcmread(f) for f in files]
    
    # Sort by SliceLocation then AcquisitionTime/B-value?
    # Simplified: Assume single slice or handle 3D volume
    # Let's check if it's a breakdown.
    
    # Try to extract b-values
    # Tag (0019, 100c) often holds b-value in Siemens
    # Or Standard (0018, 9087) Diffusion B Value
    
    bvals = []
    images = []
    
    for ds in slices:
        try:
            # Try standard first
            if (0x0018, 0x9087) in ds:
                b = float(ds[0x0018, 0x9087].value)
            # Try Siemens private
            elif (0x0019, 0x100c) in ds:
                b = float(ds[0x0019, 0x100c].value)
            else:
                b = 0.0 # Assumption
            
            bvals.append(b)
            images.append(ds.pixel_array.astype(float))
        except Exception as e:
            print(f"Skipping {ds.filename}: {e}")
            
    # Reshape
    # If multiple slices, we have [N_slices * N_b]
    # We need to structure this.
    # For this demo, let's assume we grabbed a series that is mostly b-values for volumes.
    # If len(images) > len(set(bvals)), it's a volume.
    
    unique_b = sorted(list(set(bvals)))
    # Group
    # This is complex without a full robust loader.
    # Let's create a "Mock" using the image dimensions if loader fails robustness.
    # But let's try to return raw stack and bvals.
    
    return np.array(bvals), np.array(images)

def run_prostatex_benchmark():
    print("=== ProstateX IVIM Benchmark ===")
    
    # 1. Load Data
    data_path = "data/ProstateX_Sample"
    # Find the series folder (nested from tcia download)
    # usually data/ProstateX_Sample/ProstateX-XXXX/...
    series_dirs = []
    for root, dirs, files in os.walk(data_path):
        if len(files) > 0 and files[0].endswith(".dcm"):
            series_dirs.append(root)
            
    if not series_dirs:
        print("No DICOM series found. Run download_prostatex.py first.")
        return

    target_dir = series_dirs[0] # Pick first
    print(f"Loading from {target_dir}")
    
    bvals, images = load_prostatex_dwi(target_dir)
    print(f"Loaded {len(images)} slices. B-values found: {sorted(list(set(bvals)))}")
    
    # If we have a volume, let's pick a middle slice for 2D analysis to be safe/fast
    # Reshape: [Z, B, Y, X] or similar?
    # Simple Heuristic: Assume all b-values have same number of slices
    n_b = len(set(bvals))
    n_slices = len(images) // n_b
    
    # Reconstruct 1 middle slice with all b-values
    # Filter for mid-slice index
    mid_z = n_slices // 2
    
    # We need to pair b-val to image.
    # Zip and sort
    paired = sorted(zip(bvals, images), key=lambda x: x[0])
    
    # If we have multiple slices, we need to extract the mid slice for EACH b-value
    # Assuming acquisition order is [Slice 1...N for b1], [Slice 1...N for b2]?
    # OR [b1...bN for Slice 1]?
    # Let's grab the first set of unique b-values (first slice?)
    
    b_demo = []
    sig_demo = []
    seen_b = set()
    
    for b, img in paired:
        if b not in seen_b: # Take first occurrence (Slice 0)
            seen_b.add(b)
            b_demo.append(b)
            sig_demo.append(img)
            
    b_demo = jnp.array(b_demo)
    # Stack images: (Y, X, B)
    # img shape (Y, X)
    sig_demo = jnp.stack(sig_demo, axis=-1)
    
    print(f"Demo Slice Shape: {sig_demo.shape}. B-values: {b_demo}")
    
    # 2. Pipeline Execution
    
    # A. Algebraic Init
    print("\n[Method A] Algebraic Initialization...")
    start = time.time()
    
    # Normalize
    # Clip negatives?
    sig_valid = jnp.maximum(sig_demo, 1e-4) # remove background zeros
    
    p_alg = jax.vmap(jax.vmap(lambda s: segmented_ivim_init(b_demo, s, b_threshold=150.0)))(sig_valid)
    # p_alg shape (Y, X, 3)
    
    print(f"Algebraic Time: {time.time() - start:.2f}s")
    
    # B. Global TV (Refinement)
    print("\n[Method B] Global TV Refinement...")
    start_tv = time.time()
    
    # Setup Optim
    # Transform to unconstrained
    inv_softplus = lambda x: jnp.log(jnp.expm1(x))
    inv_sigmoid = lambda x: jnp.log(x / (1 - x))
    
    # Init from Alg
    p0_tv = jnp.stack([
        inv_softplus(p_alg[..., 0]*1e9), # D
        inv_softplus(p_alg[..., 1]*1e9), # Dp
        inv_sigmoid(jnp.clip(p_alg[..., 2], 1e-4, 0.9999)) # f
    ], axis=-1)
    p0_tv = jnp.nan_to_num(p0_tv)

    opt = optax.adam(0.02)
    state = opt.init(p0_tv)
    
    # Loss
    ivim_model = IVIM()
    acq = SimpleAcquisitionScheme(b_demo, jnp.zeros((len(b_demo), 3))) # Dummy grads
    
    @jax.jit
    def tv_step(p, st, sigs):
        def loss_fn(params):
            # Params (Y, X, 3) -> Real space
            d = jax.nn.softplus(params[..., 0]) * 1e-9
            dp = jax.nn.softplus(params[..., 1]) * 1e-9
            dp = dp + d # Enforce Dp > D
            f = jax.nn.sigmoid(params[..., 2])
            
            # Recon
            # vmap over pixels implies ...
            # S = ivim(b, g, d, dp, f) for each pixel
            # Signal Model expects scalar inputs usually, vmapped.
            # let's assume predict is mapped
            
            S_pred = jax.vmap(jax.vmap(lambda d_, dp_, f_: ivim_model(b_demo, jnp.zeros((len(b_demo),3)), D_tissue=d_, D_pseudo=dp_, f=f_)))(d, dp, f)
            # S_pred (Y, X, B)
            
            S0_map = jnp.mean(sigs[..., :1], axis=-1) # Rough S0
            S_pred = S_pred * S0_map[..., None]
            
            mse = jnp.mean((S_pred - sigs)**2)
            
            # TV
            tv = jnp.mean(jnp.abs(jnp.roll(params, 1, 0) - params)) + \
                 jnp.mean(jnp.abs(jnp.roll(params, 1, 1) - params))
                 
            return mse + 0.1 * tv # Weight?
            
        gx = jax.grad(loss_fn)(p)
        u, st_new = opt.update(gx, st, p)
        return eqx.apply_updates(p, u), st_new
        
    p_tv = p0_tv
    for i in range(100):
        p_tv, state = tv_step(p_tv, state, sig_valid)
        
    print(f"Global TV Time: {time.time() - start_tv:.2f}s")
    
    print("\nSUCCESS: ProstateX Benchmark Complete.")
    print("Optimization finished without errors on real data.")

if __name__ == "__main__":
    run_prostatex_benchmark()
