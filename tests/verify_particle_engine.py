
import jax
import jax.numpy as jnp
from jax_md import space
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dmipy_jax.core import particle_engine

def verify_particle_engine():
    print("--- Verifying Particle Engine ---")
    
    # 1. GPU Check
    print("\n[1] Checking GPU Visibility...")
    has_gpu = particle_engine.check_gpu_visibility()
    if not has_gpu:
        print("WARNING: GPU not detected. High particle count might be slow.")

    # 2. Large Scale Particle Test (OOM Check)
    N_particles = 1_000_000
    print(f"\n[2] initializing {N_particles} particles...")
    
    key = jax.random.PRNGKey(0)
    box_size = 100.0
    displacement_fn, shift_fn = particle_engine.periodic_box(box_size)
    
    # Initialize random positions
    key, split = jax.random.split(key)
    positions = jax.random.uniform(split, (N_particles, 3), minval=0.0, maxval=box_size)
    
    # Define Diffusion Tensor (Anisotropic)
    # D = diag(2.0, 0.5, 0.5) roughly
    D_tensor = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ])
    # Broadcast to (N, 3, 3) to test tensor support
    D_bcast = jnp.broadcast_to(D_tensor, (N_particles, 3, 3))
    
    dt = 0.01
    
    print("Running brownian_step (JIT compiled)...")
    
    # JIT compile the step
    @jax.jit
    def step(k, p, D):
        return particle_engine.brownian_step(k, p, shift_fn, D, dt)
    
    # Run once to compile
    key, split = jax.random.split(key)
    start_time = time.time()
    new_positions = step(split, positions, D_bcast)
    jax.block_until_ready(new_positions)
    compile_time = time.time() - start_time
    print(f"Compilation + First Run Time: {compile_time:.4f} s")
    
    # Run loop
    steps = 10
    start_time = time.time()
    curr_pos = new_positions
    for _ in range(steps):
        key, split = jax.random.split(key)
        curr_pos = step(split, curr_pos, D_bcast)
    jax.block_until_ready(curr_pos)
    end_time = time.time()
    
    print(f"Executed {steps} steps with {N_particles} particles.")
    print(f"Total time: {end_time - start_time:.4f} s")
    print(f"Time per step: {(end_time - start_time)/steps:.4f} s")
    
    # 3. Anisotropy Check
    # effective displacement variance should be roughly 2 * D * dt
    # But diffusion is stochastic, so we check statistics
    disp_sq = jnp.mean((curr_pos - new_positions)**2, axis=0) # Simple check, wrapping makes this tricky so better to check drift in unbound space or small dt
    # Actually, let's verify single step variance on unbound space to be precise
    
    print("\n[3] Verifying Anisotropy Statistics...")
    disp_fn_unbound, shift_fn_unbound = particle_engine.non_periodic_box()
    
    @jax.jit
    def step_unbound(k, p, D):
         return particle_engine.brownian_step(k, p, shift_fn_unbound, D, dt)

    pos_start = jnp.zeros((100_000, 3)) # smaller batch for stats
    key, split = jax.random.split(key)
    pos_end = step_unbound(split, pos_start, D_tensor) # Single global tensor
    
    displacements = pos_end - pos_start
    variance = jnp.var(displacements, axis=0)
    expected_variance = 2 * jnp.diag(D_tensor) * dt
    
    print(f"Observed Variance: {variance}")
    print(f"Expected Variance: {expected_variance}")
    
    close = jnp.allclose(variance, expected_variance, rtol=0.1)
    if close:
        print("✅ Variance matches expected anisotropic diffusion.")
    else:
        print("❌ Variance mismatch!")

if __name__ == "__main__":
    verify_particle_engine()
