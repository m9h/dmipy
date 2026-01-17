import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.composer import compose_models
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.optimization import VoxelFitter
import time

def test_vmap_fitting():
    print("Testing Parallel (vmap) Fitting...")
    
    # 1. Define Acquisition
    bvals = jnp.concatenate([jnp.zeros(1), jnp.ones(30)*1000, jnp.ones(30)*3000])
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(61, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvecs = jnp.array(vecs)
    
    acq = JaxAcquisition(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=0.01,
        Delta=0.03
    )
    
    # 2. Composition
    ball = G1Ball()
    stick = C1Stick() 
    composite_model = compose_models([stick, ball])
    
    # 3. Create Batch of Data (Simulate a Slice)
    # Start with 1000 voxels to verify speed
    N_voxels = 1000
    print(f"Simulating {N_voxels} voxels...")
    
    # ... (Random gen same as before) ...
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    theta_gt = jnp.pi/2 + jax.random.uniform(k1, (N_voxels,), minval=-0.2, maxval=0.2)
    phi_gt   = 0.0      + jax.random.uniform(k2, (N_voxels,), minval=-0.2, maxval=0.2)
    d_par_gt = 1.7e-3   + jax.random.uniform(k3, (N_voxels,), minval=-0.1e-3, maxval=0.1e-3)
    d_iso_gt = 3.0e-3   + jax.random.uniform(k4, (N_voxels,), minval=-0.1e-3, maxval=0.1e-3)
    f_stick_gt = 0.6    + jax.random.uniform(k5, (N_voxels,), minval=-0.1, maxval=0.1)
    f_ball_gt = 1.0 - f_stick_gt
    
    # Stack params: (N, 6)
    params_gt = jnp.stack([theta_gt, phi_gt, d_par_gt, d_iso_gt, f_stick_gt, f_ball_gt], axis=1)
    
    # Pre-compute signals
    batch_predict = jax.vmap(composite_model, in_axes=(0, None))
    print("Generating synthetic signals...")
    data_batch = batch_predict(params_gt, acq)
    
    # 4. Configure Fitter
    bounds = [
        (0.0, jnp.pi), (-jnp.pi, jnp.pi), (1e-5, 5e-3), (1e-5, 5e-3), (0.0, 1.0), (0.0, 1.0)
    ]
    scales = [1.0, 1.0, 1e-3, 1e-3, 1.0, 1.0]
    
    fitter = VoxelFitter(
        composite_model, 
        bounds,
        solver_settings={'maxiter': 200, 'tol': 1e-5}, # Reduced maxiter
        scales=scales
    )
    
    init_single = jnp.array([jnp.pi/2+0.1, 0.1, 1.5e-3, 2.5e-3, 0.5, 0.5])
    init_batch = jnp.tile(init_single, (N_voxels, 1))
    
    print("Compiling vmap fitter...")
    parallel_fit = jax.vmap(fitter.fit, in_axes=(0, None, 0))
    # Warmup with small slice
    _ = parallel_fit(data_batch[:10], acq, init_batch[:10])
    print("Compilation done.")
    
    print("\nRunning Parallel Fit on GPU...")
    start = time.time()
    fitted_params_batch, states = parallel_fit(data_batch, acq, init_batch)
    fitted_params_batch.block_until_ready()
    duration = time.time() - start
    
    print(f"Batch Size: {N_voxels} voxels")
    print(f"Total Time: {duration:.4f} s")
    print(f"Throughput: {N_voxels / duration:.2f} voxels/sec")
    
    # 8. Check Accuracy (Mean Absolute Error)
    diffs = jnp.abs(fitted_params_batch - params_gt)
    mae = jnp.mean(diffs, axis=0)
    
    print("\nMean Absolute Errors:")
    names = ['theta', 'phi', 'lambda_par', 'lambda_iso', 'f_stick', 'f_ball']
    for n, e in zip(names, mae):
        print(f"{n:<15} {e:.2e}")
        
    print(f"\nMean Solver Error: {jnp.mean(states.error):.2e}")

    # Success Condition
    # Throughput > 1000 vox/s (very conservative, likely >10k)
    # Accuracy decent
    if N_voxels / duration > 500:
        print("SUCCESS: High throughput achieved.")
    else:
        print("WARNING: Low throughput.")

if __name__ == "__main__":
    test_vmap_fitting()
