
import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models import gaussian_models, stick
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.fitting.optimization import OptimistixFitter
from dmipy_jax.fitting.initialization import GlobalBruteInitializer
from dmipy_jax.acquisition import JaxAcquisition
from dmipy.data import saved_acquisition_schemes

def profile_fitting():
    print("Setting up profiling environment...")
    
    # 1. Setup Models
    ball = gaussian_models.Ball()
    stick_model = stick.Stick()
    
    # Ranges
    ball.parameter_ranges['lambda_iso'] = (1e-9, 4e-9)
    stick_model.parameter_ranges['lambda_par'] = (1e-9, 3e-9)
    
    # MCM
    jax_mcm = JaxMultiCompartmentModel(models=[ball, stick_model])
    
    # 2. Setup Data (100k voxels for profiling)
    N_voxels = 100_000
    print(f"Voxels: {N_voxels}")
    
    # ... (omitted) ...

    scale_definition_place_holder = None

    
    # Acquisition
    acq_scheme_dmipy = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme_jax = JaxAcquisition(
        bvalues=acq_scheme_dmipy.bvalues,
        gradient_directions=acq_scheme_dmipy.gradient_directions,
        delta=acq_scheme_dmipy.delta,
        Delta=acq_scheme_dmipy.Delta
    )
    
    # Generate Synthetic Data (using JAX simulation)
    np.random.seed(42)
    gt_params = {
        'partial_volume_0': np.random.uniform(0.1, 0.9, N_voxels),
        'partial_volume_1': np.zeros(N_voxels), # fill later
        'lambda_iso': np.full(N_voxels, 3e-9),
        'lambda_par': np.full(N_voxels, 1.7e-9),
        'mu': np.column_stack([
            np.random.uniform(0, np.pi, N_voxels),
            np.random.uniform(0, 2*np.pi, N_voxels)
        ])
    }
    gt_params['partial_volume_1'] = 1 - gt_params['partial_volume_0']
    
    # Convert dict to array for model_func
    # JaxMCM expects parameters in specific order
    gt_list = []
    for name in jax_mcm.parameter_names:
        val = gt_params[name]
        if jax_mcm.parameter_cardinality[name] == 1:
            gt_list.append(val[:, None])
        else:
            gt_list.append(val)
    gt_array = jnp.hstack(gt_list)
    
    print("Simulating data...")
    # Batched simulation
    data_jax = jax.vmap(jax_mcm.model_func, in_axes=(0, None))(gt_array, acq_scheme_jax)
    # Move to device and wait
    data_jax = jax.device_put(data_jax)
    jax.block_until_ready(data_jax)
    
    # 3. Profile Initialization
    print("\n--- Profiling Initialization ---")
    initializer = GlobalBruteInitializer(jax_mcm)
    key = jax.random.PRNGKey(42)
    
    start_time = time.time()
    candidates = initializer.generate_random_grid(n_samples=2000, key=key)
    print(f"Candidate generation: {time.time() - start_time:.4f} s")
    
    # Precompute candidate predictions
    print("Compiling Simulator...")
    t0 = time.time()
    simulator = jax.jit(jax.vmap(jax_mcm.model_func, in_axes=(0, None)))
    # Trigger compile
    _ = simulator(candidates[:2], acq_scheme_jax)
    print(f"Simulator Compile Time: {time.time() - t0:.4f} s")
    
    t0 = time.time()
    candidate_predictions = simulator(candidates, acq_scheme_jax)
    jax.block_until_ready(candidate_predictions)
    print(f"Simulator Exec Time: {time.time() - t0:.4f} s")
    
    # Select best
    print("Compiling Selector...")
    t0 = time.time()
    selector = jax.jit(jax.vmap(initializer.select_best_candidate, in_axes=(0, None, None)))
    # Trigger compile (shim)
    _ = selector(data_jax[:2], candidate_predictions, candidates)
    print(f"Selector Compile Time: {time.time() - t0:.4f} s")
    
    t0 = time.time()
    init_params = selector(data_jax, candidate_predictions, candidates)
    jax.block_until_ready(init_params)
    init_exec_time = time.time() - t0
    print(f"Selector Exec Time: {init_exec_time:.4f} s")
    
    init_time = time.time() - start_time
    print(f"Total Initialization Time: {init_time:.4f} s")
    
    # Check Init Quality (MSE)
    init_preds = simulator(init_params, acq_scheme_jax)
    init_mse = jnp.mean((data_jax - init_preds)**2)
    print(f"Initialization Mean MSE: {init_mse:.6e}")
    
    # 4. Profile Fitting
    print("\n--- Profiling Fitting (Optimistix) ---")
    
    # Prepare Fitter
    # Replicate scaling logic from modeling_framework
    flat_ranges = []
    scales_list = []
    for name in jax_mcm.parameter_names:
        card = jax_mcm.parameter_cardinality[name]
        rng = jax_mcm.parameter_ranges[name]
        
        current_scales = []
        if card == 1:
            flat_ranges.append(rng)
            low, high = rng
            if not np.isinf(high) and (high != 0):
                s = high
            elif not np.isinf(low) and (low != 0):
                s = low
            else:
                s = 1.0
            current_scales.append(s)
        else:
            if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], (int, float)):
                 flat_ranges.extend([rng] * card)
                 low, high = rng
                 s = high if not np.isinf(high) and high!=0 else 1.0
                 current_scales.extend([s] * card)
            else:
                 flat_ranges.extend(rng)
                 for r in rng:
                     l, h = r
                     s = h if not np.isinf(h) and h!=0 else 1.0
                     current_scales.append(s)
        scales_list.extend(current_scales)
    scales = jnp.array(scales_list)
    print(f"Scales: {scales}")

    from dmipy_jax.fitting.optimization import CustomLMFitter
    print("SWITCHING TO CUSTOM LM FITTER")
    fitter = CustomLMFitter(jax_mcm.model_func, flat_ranges, scales=scales)
    
    # Compile fit
    print("Compiling Fitter...")
    fit_vmapped = jax.jit(jax.vmap(fitter.fit, in_axes=(0, None, 0)))
    
    BATCH_SIZE = 5000
    
    # Warmup with one batch
    print(f"Warming up with BATCH_SIZE={BATCH_SIZE}...")
    _ = fit_vmapped(data_jax[:BATCH_SIZE], acq_scheme_jax, init_params[:BATCH_SIZE])
    
    print("Running Chunked Fit...")
    start_time = time.time()
    
    all_steps = []
    
    # Process in chunks
    N_batches = int(np.ceil(N_voxels / BATCH_SIZE))
    for i in range(N_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, N_voxels)
        if end_idx - start_idx != BATCH_SIZE:
             continue
             
        batch_data = data_jax[start_idx:end_idx]
        batch_init = init_params[start_idx:end_idx]
        
        # Fit
        # CustomLMFitter returns (params, steps)
        fitted_batch, steps_batch = fit_vmapped(batch_data, acq_scheme_jax, batch_init)
        
        # Block to ensure timing
        fitted_batch.block_until_ready()
        
        # Collect stats
        all_steps.append(np.array(steps_batch))
        
    fit_time = time.time() - start_time
    print(f"Fit Time: {fit_time:.4f} s")
    print(f"Throughput: {N_voxels / fit_time:.2f} voxels/sec")
    
    # 5. Inspect Stats
    print("\n--- Solver Statistics ---")
    steps = np.concatenate(all_steps)
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Max Steps: {np.max(steps)}")
    print(f"Min Steps: {np.min(steps)}")
    
    # Breakdown
    print("Step Histogram:")
    hist, bins = np.histogram(steps, bins=10)
    for h, b in zip(hist, bins):
        print(f"  {b:.1f}: {h}")
        
if __name__ == "__main__":
    profile_fitting()
