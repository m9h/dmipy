
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dmipy_jax.models.epg import JAXEPG

def benchmark_ssfp_performance():
    print("--> Architect: Starting JAX EPG Benchmark...")
    
    # 1. Setup Synthetic Data (representing a large slice/volume)
    # WAND High-Res Volume: approx 200x200x120 ~ 5M voxels?
    # Let's benchmark 1 Million voxels.
    N_VOXELS = 1_000_000
    print(f"--> Workload: {N_VOXELS} voxels (Simulating 1 Volume)")
    
    # Random tissue parameters
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    T1 = jax.random.uniform(k1, (N_VOXELS,), minval=500.0, maxval=1500.0) # ms
    T2 = jax.random.uniform(k2, (N_VOXELS,), minval=40.0, maxval=200.0)   # ms
    OffRes = jax.random.uniform(k3, (N_VOXELS,), minval=-3.14, maxval=3.14) # radians
    
    # Protocol
    TR = 5.0 # ms
    Alpha = jnp.deg2rad(30.0)
    Phi_Cycle = jnp.deg2rad(180.0) # bSSFP standard
    
    # 2. Define Vmapped Function
    # We map over the voxel dimension (0) for T1, T2, OffRes
    # TR, Alpha, Phi, N_pulses are broadcasted/static
    
    def single_voxel_sim(t1, t2, off_res):
        return JAXEPG.simulate_bssfp(t1, t2, TR, Alpha, off_resonance=off_res, 
                                     phase_cycling=Phi_Cycle, N_pulses=100)

    # Vmap it
    batch_sim = jax.jit(jax.vmap(single_voxel_sim))
    
    # 3. Compilation / Warmup
    print("--> Compiling JAX Kernel on GPU...")
    start_compile = time.time()
    _ = batch_sim(T1[:10], T2[:10], OffRes[:10]).block_until_ready()
    end_compile = time.time()
    print(f"    Compilation Time: {end_compile - start_compile:.4f} s")
    
    # 4. Execution Timing
    print("--> Running Benchmark...")
    start_run = time.time()
    final_output = batch_sim(T1, T2, OffRes).block_until_ready()
    end_run = time.time()
    
    duration = end_run - start_run
    throughput = N_VOXELS / duration
    
    print(f"    Execution Time:   {duration:.4f} s")
    print(f"    Throughput:       {throughput:,.0f} voxels/sec")
    print(f"    Est. Whole Brain (5M vox): {5_000_000 / throughput:.4f} s")
    
    # 5. Accuracy Check (Sanity)
    if jnp.any(jnp.isnan(final_output)):
        print("!! WARNING: NaNs detected in output !!")
    else:
        print("    Numerical Stability: PASS (No NaNs)")

    return throughput

if __name__ == "__main__":
    benchmark_ssfp_performance()
