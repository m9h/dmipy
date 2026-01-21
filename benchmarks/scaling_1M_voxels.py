import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
from dmipy_jax.composer import compose_models
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.optimization import OptimistixFitter

def run_benchmark(n_voxels: int, use_test_mode: bool = False, solver_name: str = "lm"):
    print(f"Starting Scaling Benchmark with {n_voxels} voxels using {solver_name.upper()}...")
    
    # 1. Define Acquisition
    # Standard shell acquisition
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
    
    # 2. Composition (Ball and Stick)
    ball = G1Ball()
    stick = C1Stick() 
    composite_model = compose_models([stick, ball])
    
    # 3. Create Batch of Data (Simulate)
    print(f"Simulating {n_voxels} synthetic voxels...")
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    theta_gt = jnp.pi/2 + jax.random.uniform(k1, (n_voxels,), minval=-0.2, maxval=0.2)
    phi_gt   = 0.0      + jax.random.uniform(k2, (n_voxels,), minval=-0.2, maxval=0.2)
    d_par_gt = 1.7e-3   + jax.random.uniform(k3, (n_voxels,), minval=-0.1e-3, maxval=0.1e-3)
    d_iso_gt = 3.0e-3   + jax.random.uniform(k4, (n_voxels,), minval=-0.1e-3, maxval=0.1e-3)
    f_stick_gt = 0.6    + jax.random.uniform(k5, (n_voxels,), minval=-0.1, maxval=0.1)
    f_ball_gt = 1.0 - f_stick_gt
    
    # Stack params: (N, 6)
    params_gt = jnp.stack([theta_gt, phi_gt, d_par_gt, d_iso_gt, f_stick_gt, f_ball_gt], axis=1)
    
    # Pre-compute signals
    # We use vmap over the model for simulation
    batch_predict = jax.vmap(composite_model, in_axes=(0, None))
    data_batch = batch_predict(params_gt, acq)
    
    # 4. Configure Fitter
    # Bounds for scaling (OptimistixFitter uses these to set scales usually, or we pass scales explicitly)
    bounds = [
        (0.0, jnp.pi), (-jnp.pi, jnp.pi), (1e-5, 5e-3), (1e-5, 5e-3), (0.0, 1.0), (0.0, 1.0)
    ]
    scales = [1.0, 1.0, 1e-3, 1e-3, 1.0, 1.0]

    if solver_name == "lm":
        # Solver settings for Levenberg-Marquardt
        solver_settings = {
            'rtol': 1e-5, 
            'atol': 1e-5
        }
        
        fitter = OptimistixFitter(
            composite_model, 
            bounds, 
            solver_settings=solver_settings,
            scales=scales
        )
    else:
        # LBFGS via VoxelFitter (JaxOpt)
        from dmipy_jax.fitting.optimization import VoxelFitter
        fitter = VoxelFitter(
            composite_model,
            bounds,
            solver_settings={'maxiter': 200, 'tol': 1e-5},
            scales=scales
        )
    
    # Initial guess
    init_single = jnp.array([jnp.pi/2+0.1, 0.1, 1.5e-3, 2.5e-3, 0.5, 0.5])
    init_batch = jnp.tile(init_single, (n_voxels, 1))
    
    # 5. Compilation Phase
    print("Compiling JIT (Warmup)...")
    # vmap the fit method: (data, acquisition, init_params) -> (fitted_params, status)
    parallel_fit = jax.vmap(fitter.fit, in_axes=(0, None, 0))
    
    # Warmup on a tiny slice to trigger compilation
    warmup_size = min(100, n_voxels)
    start_compile = time.time()
    _ = parallel_fit(data_batch[:warmup_size], acq, init_batch[:warmup_size])
    # Block to ensure compilation finishes
    _[0].block_until_ready()
    compile_time = time.time() - start_compile
    print(f"Compilation Time: {compile_time:.4f} s")
    
    # 6. Execution Phase
    print("Running Measurement Phase...")
    
    # Process in chunks to avoid single massive kernel launch/OOM issues and show progress
    chunk_size = 100_000
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    
    start_exec = time.time()
    
    total_processed = 0
    all_fitted_params = []
    
    print(f"Processing {n_voxels} voxels in {n_chunks} chunks of {chunk_size}...")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        current_batch_size = end_idx - start_idx
        
        # Slice data
        chunk_data = data_batch[start_idx:end_idx]
        chunk_init = init_batch[start_idx:end_idx]
        
        # Run fit
        chunk_params, _ = parallel_fit(chunk_data, acq, chunk_init)
        chunk_params.block_until_ready()
        
        if use_test_mode:
            all_fitted_params.append(chunk_params)
            
        total_processed += current_batch_size
        print(f"  Chunk {i+1}/{n_chunks} done. ({total_processed}/{n_voxels})")

    exec_time = time.time() - start_exec
    
    if use_test_mode:
        fitted_params_batch = jnp.concatenate(all_fitted_params, axis=0)
    else:
        # Just use last chunk for dummy var if needed
        fitted_params_batch = chunk_params 
    
    throughput = n_voxels / exec_time
    print(f"Execution Time: {exec_time:.4f} s")
    print(f"Throughput: {throughput:.2f} Voxels/Second")
    
    # 7. Verification Results (Optional check)
    if use_test_mode:
        mae = jnp.mean(jnp.abs(fitted_params_batch - params_gt), axis=0)
        print("\nTest Mode Verification (MAE):")
        names = ['theta', 'phi', 'd_par', 'd_iso', 'f_stick', 'f_ball']
        for n, e in zip(names, mae):
            print(f"  {n}: {e:.2e}")
        
    print(f"\nReport for {n_voxels} voxels:")
    print(f"  Compile: {compile_time:.4f} s")
    print(f"  Exec:    {exec_time:.4f} s")
    print(f"  Rate:    {throughput:.0f} v/s")

def main():
    parser = argparse.ArgumentParser(description="Scaling Benchmark for JAX Fitting")
    parser.add_argument("--test", action="store_true", help="Run small test version (10k voxels)")
    parser.add_argument("--solver", type=str, default="lm", choices=["lm", "lbfgs"], help="Choose solver: 'lm' (Optimistix) or 'lbfgs' (JaxOpt)")
    args = parser.parse_args()
    
    if args.test:
        N = 10_000
    else:
        N = 1_000_000
        
    try:
        run_benchmark(N, use_test_mode=args.test, solver_name=args.solver)
        print("\nSUCCESS")
    except Exception as e:
        print(f"\nFAILED with error: {e}")
        raise

if __name__ == "__main__":
    main()
