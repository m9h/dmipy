
import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.plane_models import PlaneStejskalTanner, PlaneCallaghan
from dmipy_jax.signal_models.sandi import SphereGPD
from dmipy_jax.acquisition import JaxAcquisition

def benchmark_plane_models():
    print("\nBenchmarking Plane Models...")
    print("=" * 60)
    
    # Setup
    N_VOXELS = 10000
    bvals = jnp.array([1000.0, 2000.0, 3000.0]) # 3-shell
    # Expand bvals to mock directions: 60 dirs
    bvals = jnp.repeat(bvals, 20)
    gradients = jnp.ones((60, 3))
    gradients = gradients / jnp.linalg.norm(gradients, axis=1, keepdims=True)
    
    # Timings
    big_delta = 0.04 # 40ms
    small_delta = 0.01 # 10ms
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=gradients, delta=small_delta, Delta=big_delta)
    
    # Models
    model_st = PlaneStejskalTanner()
    model_cal = PlaneCallaghan(number_of_roots=20)
    
    # Parameters
    diameter = jnp.ones(N_VOXELS) * 5e-6
    diff_const = jnp.ones(N_VOXELS) * 2e-9
    
    # --- Benchmark Stejskal-Tanner ---
    @jax.jit
    def run_st(d):
        return jax.vmap(lambda x: model_st(acq.bvalues, acq.gradient_directions, diameter=x, big_delta=big_delta, small_delta=small_delta))(d)
    
    # Warmup
    print("Warming up Stejskal-Tanner...")
    _ = run_st(diameter[:10]).block_until_ready()
    
    # Run
    t0 = time.perf_counter()
    _ = run_st(diameter).block_until_ready()
    t_st = time.perf_counter() - t0
    print(f"Stejskal-Tanner: {t_st:.4f} s ({N_VOXELS/t_st:.0f} vox/s)")
    
    # --- Benchmark Callaghan (Infinite Sum) ---
    @jax.jit
    def run_cal(d, Dc):
        return jax.vmap(lambda x, y: model_cal(acq.bvalues, acq.gradient_directions, diameter=x, diffusion_constant=y, big_delta=big_delta, small_delta=small_delta))(d, Dc)
        
    # Warmup
    print("Warming up Callaghan...")
    _ = run_cal(diameter[:10], diff_const[:10]).block_until_ready()
    
    # Run
    t0 = time.perf_counter()
    _ = run_cal(diameter, diff_const).block_until_ready()
    t_cal = time.perf_counter() - t0
    print(f"Callaghan (20 roots): {t_cal:.4f} s ({N_VOXELS/t_cal:.0f} vox/s)")
    
    print(f"Slowdown Factor: {t_cal/t_st:.2f}x (Expected due to loop)")

def benchmark_sandi_component():
    print("\nBenchmarking SANDI Component (SphereGPD)...")
    print("=" * 60)
    
    # Setup
    N_VOXELS = 10000
    bvals = jnp.ones(60) * 2000.0
    gradients = jnp.ones((60, 3))
    
    # Timings must be in acquisition for SphereGPD if passed implicitly, or explicit
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=gradients, delta=0.01, Delta=0.04)
    
    model = SphereGPD()
    
    diameter = jnp.ones(N_VOXELS) * 8e-6
    diff_const = jnp.ones(N_VOXELS) * 3e-9
    
    @jax.jit
    def run_gpd(d, D):
        # Pass acquisition object to get delta/Delta
        return jax.vmap(lambda x, y: model(acq.bvalues, acq.gradient_directions, acquisition=acq, diameter=x, diffusion_constant=y))(d, D)
        
    # Warmup
    print("Warming up SphereGPD...")
    _ = run_gpd(diameter[:10], diff_const[:10]).block_until_ready()
    
    # Run
    t0 = time.perf_counter()
    _ = run_gpd(diameter, diff_const).block_until_ready()
    t_gpd = time.perf_counter() - t0
    print(f"SphereGPD (Root Sum): {t_gpd:.4f} s ({N_VOXELS/t_gpd:.0f} vox/s)")


if __name__ == "__main__":
    benchmark_plane_models()
    benchmark_sandi_component()
