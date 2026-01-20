
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dmipy_jax.io.datasets import load_stanford_hardi, load_sherbrooke_3shell, load_bigmac_mri
from dmipy_jax.signal_models import cylinder_models, sphere_models, Stick, Ball, Zeppelin
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def benchmark_dataset(dataset_name, load_func, voxel_slice=None):
    print(f"\n--- Benchmarking {dataset_name} ---")
    
    # 1. Load Data
    t0 = time.time()
    data, scheme = load_func(voxel_slice=voxel_slice)
    print(f"Data Loaded: {data.shape} voxels, {len(scheme.bvalues)} measurements. Time: {time.time()-t0:.2f}s")
    
    # Flatten non-spatial dimensions for easier reporting (voxels, measurements)
    n_voxels = np.prod(data.shape[:-1])
    print(f"Total Voxels: {n_voxels}")
    
    # 2. Define Models
    # Model A: Diffusion Tensor (using Cylinder/Zeppelin approx or standard DTI?)
    # Let's use a meaningful biophysical model: NODDI-like (Stick + Zeppelin + Sphere)
    # Stick (Intra-neurite)
    stick = Stick()
    # Zeppelin (Extra-neurite)
    zeppelin = Zeppelin()
    # Ball (CSF) - isotropic
    ball = Ball() 
    
    # Model 1: Ball (Simple)
    mcm_ball = JaxMultiCompartmentModel([ball])
    
    # Model 2: Stick + Zeppelin (2-compartment)
    mcm_noddi_simple = JaxMultiCompartmentModel([stick, zeppelin])
    
    # 3. Benchmark Fitting
    models = [("S0Ball", mcm_ball), ("Stick+Zeppelin", mcm_noddi_simple)]
    
    results = []
    
    for name, model in models:
        print(f"Fitting {name}...")
        
        # JIT Compile Fit (First Run)
        # We assume fit method handles compilation.
        # To measure overhead, we might want to separate it, but fit() usually does it.
        # We'll measure the first batch vs subsequent batches if we chunked, 
        # but here we pass the whole dataset (or slice).
        
        # Reshape data to (N_vox, N_meas) for fit function if it expects 2D?
        # The new fit API handles arbitrary shapes? Let's check.
        # fit(acquisition, data) usually iterates or vmaps.
        
        start_fit = time.time()
        # Use OptimistixFitter if available in fit().
        # We need to verify the fit implementation in modeling_framework.
        # Assuming model.fit(scheme, data) works.
        
        fitted_params = model.fit(scheme, data)
        # Block until ready
        # fitted_params is usually a dict of arrays.
        # Access one to force sync.
        block_val = list(fitted_params.values())[0] if fitted_params else None
        if hasattr(block_val, 'block_until_ready'):
             block_val.block_until_ready()
             
        end_fit = time.time()
        duration = end_fit - start_fit
        
        print(f"  Duration: {duration:.4f} s")
        print(f"  Throughput: {n_voxels / duration:.2f} voxels/s")
        
        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "Voxels": n_voxels,
            "Time (s)": duration,
            "Voxels/s": n_voxels / duration
        })
        
    return results

def main():
    print("=== Dmipy-JAX Real Data Benchmark ===")
    
    # Limit slice for rapid testing? Or run full?
    # Full fit might take a while on CPU. On GPU it's fast.
    # Let's take a substantial slice: center 10 slices z=30:40
    # Standard images are usually ~80x80x50.
    
    slice_stanford = (slice(20, 60), slice(20, 60), slice(30, 40)) # 40x40x10 = 16000 voxels
    slice_sherbrooke = (slice(20, 60), slice(20, 60), slice(30, 40))

    try:
        res_stanford = benchmark_dataset("Stanford HARDI", load_stanford_hardi, voxel_slice=slice_stanford)
    except ImportError as e:
        print(f"Skipping Stanford: {e}")
        res_stanford = []

    try:
        res_sherbrooke = benchmark_dataset("Sherbrooke 3-Shell", load_sherbrooke_3shell, voxel_slice=slice_sherbrooke)
    except ImportError as e:
        print(f"Skipping Sherbrooke: {e}")
        res_sherbrooke = []
        
    # Wrapper for BigMac to match standard signature (data, scheme)
    def load_bigmac_wrapper(voxel_slice=None):
        res = load_bigmac_mri(voxel_slice=voxel_slice)
        return res['dwi'], res['scheme']

    try:
        # Use a slice for BigMac too, assuming it's large.
        res_bigmac = benchmark_dataset("BigMac", load_bigmac_wrapper, voxel_slice=slice_stanford)
    except (ImportError, FileNotFoundError) as e:
        print(f"Skipping BigMac: {e}")
        res_bigmac = []

    all_results = res_stanford + res_sherbrooke + res_bigmac
    
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n--- Summary Results ---")
        print(df.to_markdown(index=False))
        # Save?
        # df.to_csv("benchmark_real_data_results.csv")

if __name__ == "__main__":
    main()
