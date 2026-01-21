
import jax
import jax.numpy as jnp
import time
import sys
import os
from dmipy_jax.inverse.amico import AMICOSolver

# Dummy model for dictionary generation
def dummy_stick(params, acquisition):
    # simple linear model: signal = diff * bval
    bvals = acquisition['bvals']
    D = params['diffusivity']
    # broadcasting D vs bvals
    # D: scalar or (N_atoms,) -- wait, dictionary generation passes values
    # In generate_kernels wrapper: p_dict = {k:v}. v is scalar.
    return jnp.exp(-bvals * D)

def run_benchmark():
    print(f"JAX Platform: {jax.devices()[0].platform.upper()}")
    
    # 1. Setup Acquisition and Dictionary
    # 100 measurements
    bvals = jnp.linspace(0, 3000, 100)
    acquisition = {'bvals': bvals}
    
    # Dictionary Size: 2000 atoms
    # This generates a [100, 2000] matrix. 
    # Size: 100 * 2000 * 4 bytes = 800 KB (Small)
    # The bottleneck is the voxel count.
    N_atoms = 2000
    ds = jnp.linspace(1e-4, 3e-3, N_atoms)
    dict_params = {'diffusivity': ds}
    
    print(f"Initializing Solver with {N_atoms} atoms and {len(bvals)} measurements...")
    solver = AMICOSolver(dummy_stick, acquisition, dict_params)
    
    # Pre-compile the fit function
    # We use a dummy small batch to compile
    print("Compiling fit...")
    dummy_data = jnp.zeros((10, 100))
    start_compile = time.time()
    _ = solver.fit(dummy_data, lambda_reg=0.0, constrained=True).block_until_ready()
    print(f"Compilation finished in {time.time() - start_compile:.2f}s")
    
    # 2. Run Stress Test
    # Target: 1 Million Voxels
    # Input Data: 1M * 100 * 4 bytes = 400 MB.
    # AtY: 1M * 2000 * 4 bytes = 8 GB.
    # X, Z, U: 3 * 8 GB = 24 GB.
    # Total VRAM needed ~ 32 GB + overhead.
    # This should OOM on standard 16GB/24GB GPUs.
    
    N_voxels = 1_000_000
    print(f"\nAttempting to fit {N_voxels} voxels (Requires ~32GB VRAM for variables)...")
    
    # Generate random data
    key = jax.random.PRNGKey(0)
    # create data on CPU first to avoid OOM during allocations if possible, then move? 
    # Or just create directly.
    # To test solver memory, let's try to creating it.
    
    try:
        # data = jax.random.normal(key, (N_voxels, 100)) 
        # Actually random noise is fine.
        # But let's build valid signals to overlap somewhat
        # Just random weights
        true_w = jax.random.uniform(key, (N_voxels, N_atoms), minval=0.0, maxval=0.1)
        # Sparse?
        # A bit too big to simulate simply? 1M x 2000 is 8GB.
        # Let's just create random data (N_voxels, 100) -> 400MB. Easy.
        data = jax.random.uniform(key, (N_voxels, 100))
        
        print(f"Data created. Shape: {data.shape}")
        
        start_fit = time.time()
        res = solver.fit(data, lambda_reg=0.001, constrained=True)
        res.block_until_ready()
        end_fit = time.time()
        
        print(f"SUCCESS! Fit {N_voxels} voxels in {end_fit - start_fit:.2f}s")
        print(f"Throughput: {N_voxels / (end_fit - start_fit):.2f} voxels/s")
        
    except Exception as e:
        print(f"\nFAILED with error: {e}")
        print("This confirms the Memory Bottleneck.")

if __name__ == "__main__":
    run_benchmark()
