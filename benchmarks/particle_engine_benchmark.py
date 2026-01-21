
import time
import jax
import jax.numpy as jnp
from jax_md import space
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dmipy_jax.core import particle_engine

def run_benchmark():
    print("=========================================================")
    print("       JAX-MD Particle Engine: DGX Spark Benchmark       ")
    print("=========================================================")
    
    # Check Devices
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    cpu_devices = [d for d in devices if d.platform == 'cpu']
    
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Available GPUs: {len(gpu_devices)}")
    if gpu_devices:
        print(f"  - {gpu_devices[0].device_kind}")
    else:
        print("  - None (Running on CPU)")
    print(f"Available CPUs: {len(cpu_devices)}\n")

    # Parameters
    # Scaling to 50M for DGX
    particle_counts = [100_000, 1_000_000, 10_000_000, 50_000_000] 
    steps_per_run = 100
    box_size = 100.0
    dt = 0.01
    
    # Diffusion Tensor
    D_tensor = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ])
    D_global = D_tensor
    displacement_fn, shift_fn = particle_engine.periodic_box(box_size)

    # Simulation Inner Loop
    def simulation_loop(carrier, _):
        k, p = carrier
        k, split = jax.random.split(k)
        new_p = particle_engine.brownian_step(split, p, shift_fn, D_global, dt)
        return (k, new_p), None

    # JIT Compiled Functions
    # GPU Version
    @jax.jit
    def run_sim_gpu(key, pos):
        (final_key, final_pos), _ = jax.lax.scan(simulation_loop, (key, pos), None, length=steps_per_run)
        return final_pos

    # CPU Version (forced backend)
    # We define it separately to ensure JIT targets CPU
    @jax.jit(backend='cpu')
    def run_sim_cpu(key, pos):
        (final_key, final_pos), _ = jax.lax.scan(simulation_loop, (key, pos), None, length=steps_per_run)
        return final_pos

    results = []

    for N in particle_counts:
        print(f"--- Benchmarking N = {N:,} ---")
        
        # Init Data
        key = jax.random.PRNGKey(0)
        key, split = jax.random.split(key)
        # Init on default device (likely GPU)
        positions = jax.random.uniform(split, (N, 3), minval=0.0, maxval=box_size)
        
        # ---------------------------
        # GPU Run
        # ---------------------------
        # Compilation
        print("  [GPU] Compiling...", end="", flush=True)
        t0 = time.time()
        key, split = jax.random.split(key)
        _ = run_sim_gpu(split, positions).block_until_ready()
        t1 = time.time()
        gpu_compile = t1 - t0
        print(f" Done ({gpu_compile:.4f}s)")
        
        # Execution
        print(f"  [GPU] Running {steps_per_run} steps...", end="", flush=True)
        t0 = time.time()
        final_pos_gpu = run_sim_gpu(split, positions).block_until_ready()
        t1 = time.time()
        gpu_exec = t1 - t0
        print(f" Done ({gpu_exec:.4f}s)")
        
        gpu_updates_per_sec = (N * steps_per_run) / gpu_exec

        # ---------------------------
        # CPU Baseline (Only for N <= 1M to avoid waiting forever)
        # ---------------------------
        cpu_exec = float('nan')
        speedup = float('nan')
        
        if N <= 1_000_000:
            # Move data to CPU for fair CPU benchmark (avoid transfer overhead dominance? 
            # actually JAX handles this, but explicit is good)
            pos_cpu = jax.device_put(positions, jax.devices('cpu')[0])
            
            print("  [CPU] Compiling...", end="", flush=True)
            t0 = time.time()
            key, split = jax.random.split(key)
            _ = run_sim_cpu(split, pos_cpu).block_until_ready() # Compilation run
            print(" Done")
            
            print(f"  [CPU] Running {steps_per_run} steps...", end="", flush=True)
            t0 = time.time()
            _ = run_sim_cpu(split, pos_cpu).block_until_ready()
            t1 = time.time()
            cpu_exec = t1 - t0
            print(f" Done ({cpu_exec:.4f}s)")
            
            speedup = cpu_exec / gpu_exec
        else:
            print("  [CPU] Skipped (Too large)")

        
        results.append({
            "N_particles": N,
            "GPU_Time_s": gpu_exec,
            "CPU_Time_s": cpu_exec,
            "Speedup_Factor": speedup,
            "M_Updates/sec": gpu_updates_per_sec / 1e6
        })
        
    # Output
    df = pd.DataFrame(results)
    print("\n\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    run_benchmark()
