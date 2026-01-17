import numpy as np
import jax.numpy as jnp
from dipy.data import read_sherbrooke_3shell
from dmipy_jax.signal_models import g2_zeppelin
import time

def run_benchmark():
    print("Loading Sherbrooke 3-Shell Dataset...")
    img, gtab = read_sherbrooke_3shell()
    data = img.get_fdata()
    bvals = jnp.array(gtab.bvals)
    bvecs = jnp.array(gtab.bvecs)
    
    print(f"Data Shape: {data.shape}")
    print(f"B-values: {np.unique(bvals)}")
    print(f"Number of gradients: {len(bvals)}")
    
    # Select a voxel with signal (avoid background)
    # Center of image usually has brain
    x, y, z = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    voxel_signal = data[x, y, z, :]
    
    # Normalize signal (S/S0)
    # Find b0 indices
    b0_mask = bvals < 10
    s0 = jnp.mean(voxel_signal[b0_mask])
    s_norm = voxel_signal / s0
    
    print(f"Selected Voxel at ({x}, {y}, {z})")
    print(f"S0: {s0:.2f}")
    
    # Simulation: Predict signal for this acquisition scheme using Zeppelin
    # We will just manually set some realistic parameters to see if the model runs efficiently
    
    # Parameters
    mu = jnp.array([1.0, 0.0, 0.0]) # Assume fiber along X
    lambda_par = 1.7e-3 # 1.7 um^2/ms
    lambda_perp = 0.2e-3 # 0.2 um^2/ms
    
    print("\nRunning JAX Forward Prediction (Zeppelin)...")
    start_time = time.time()
    
    # Run twice to gauge compilation vs execution
    # First run (Compilation)
    s_pred = g2_zeppelin(bvals, bvecs, mu, lambda_par, lambda_perp)
    s_pred.block_until_ready()
    compile_time = time.time() - start_time
    print(f"First Call (Compile + Exec): {compile_time*1000:.2f} ms")
    
    # Second run (Execution)
    start_time = time.time()
    s_pred = g2_zeppelin(bvals, bvecs, mu, lambda_par, lambda_perp)
    s_pred.block_until_ready()
    exec_time = time.time() - start_time
    print(f"Second Call (Execution): {exec_time*1000:.2f} ms")
    
    print(f"Predicted Signal Mean: {jnp.mean(s_pred):.4f}")
    print("Benchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
