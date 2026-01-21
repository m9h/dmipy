
import time
import numpy as np
import jax
import jax.numpy as jnp
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy_jax.signal_models import gaussian_models as jax_gaussian
from dmipy_jax.signal_models import stick as jax_stick
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.optimization import CustomLMFitter
# from sklearn.metrics import r2_score, mean_squared_error

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def benchmark_parity():
    print("One-to-One Parity Benchmark: dmipy (CPU) vs dmipy-jax (GPU)")
    
    # 1. Setup Models
    # Standard dmipy
    ball_cpu = gaussian_models.G1Ball()
    stick_cpu = cylinder_models.C1Stick()
    mcm_cpu = MultiCompartmentModel(models=[ball_cpu, stick_cpu])
    
    # Jax dmipy
    ball_jax = jax_gaussian.Ball()
    stick_jax = jax_stick.Stick()
    
    # Align ranges
    ball_jax.parameter_ranges['lambda_iso'] = (0, 3e-9) # dmipy defaults?
    stick_jax.parameter_ranges['lambda_par'] = (0, 3e-9)
    # Actually dmipy often defaults to wider ranges or None.
    # Let's check ranges to ensure fair comparison
    # G1Ball: lambda_iso [0, 3e-9] in dmipy usually
    # C1Stick: lambda_par [0, 3e-9]
    
    jax_mcm = JaxMultiCompartmentModel(models=[ball_jax, stick_jax])
    print("JAX MCM Param Names:", jax_mcm.parameter_names)
    
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    
    jax_acq = JaxAcquisition(
        bvalues=acq_scheme.bvalues,
        gradient_directions=acq_scheme.gradient_directions,
        delta=acq_scheme.delta,
        Delta=acq_scheme.Delta
    )

    # 2. Data
    N_voxels = 2000 # Scaling up for final check
    print(f"Simulating {N_voxels} voxels...")
    
    np.random.seed(42)
    # Ground Truth
    gt_params = {}
    gt_params['partial_volume_0'] = np.random.uniform(0.1, 0.9, N_voxels)
    gt_params['partial_volume_1'] = 1 - gt_params['partial_volume_0']
    gt_params['G1Ball_1_lambda_iso'] = np.full(N_voxels, 2.5e-9)
    gt_params['C1Stick_1_lambda_par'] = np.full(N_voxels, 1.7e-9)
    gt_params['C1Stick_1_mu'] = np.column_stack([
        np.random.uniform(0, np.pi, N_voxels),
        np.random.uniform(0, 2*np.pi, N_voxels)
    ])
    
    # Simulate CPU data
    data_cpu = mcm_cpu.simulate_signal(acq_scheme, gt_params)
    # Add noise?
    # Rician noise
    data_noisy = data_cpu + np.random.normal(0, 0.02, data_cpu.shape) # Simple Gaussian for now
    
    # 3. Fit CPU
    print("\n--- Running CPU Fit (dmipy) ---")
    t0 = time.time()
    # use_parallel=False for easier benchmarking control, or True if available
    # dmipy fitting is usually: fit(scheme, data, mask)
    # We fit without mask
    # Pathos missing, running serial
    fit_results_cpu = mcm_cpu.fit(acq_scheme, data_noisy, use_parallel_processing=False) 
    cpu_time = time.time() - t0
    print(f"CPU Time: {cpu_time:.4f} s")
    print(f"CPU Throughput: {N_voxels/cpu_time:.2f} voxels/s")
    
    # Extract CPU params
    fitted_pv_cpu = fit_results_cpu.fitted_parameters['partial_volume_0']
    fitted_diff_cpu = fit_results_cpu.fitted_parameters['G1Ball_1_lambda_iso']
    
    # 4. Fit GPU
    print("\n--- Running GPU Fit (dmipy-jax) ---")
    data_jax = jnp.array(data_noisy)
    
    # Init
    # For fair comparison, use simple brute force or similar init?
    # dmipy uses MIX optimization (Mix of brute and solver).
    # We will use our GlobalBrute + CustomLM.
    
    # Prepare ranges/scales
    flat_ranges = []
    scales_list = []
    for name in jax_mcm.parameter_names:
        rng = jax_mcm.parameter_ranges[name]
        card = jax_mcm.parameter_cardinality[name]
        # range logic
        if card == 1:
             l, h = rng
             s = h if h!=0 and not np.isinf(h) else 1.0
             scales_list.append(s)
             flat_ranges.append(rng)
        else:
             flat_ranges.extend(rng) # Handle tuple list
             for r in rng:
                  l,h=r
                  s = h if h!=0 and not np.isinf(h) else 1.0
                  scales_list.append(s)
    
    scale_definition_place_holder = None

    scales = jnp.array(scales_list)
    custom_fitter = CustomLMFitter(jax_mcm.model_func, flat_ranges, scales=scales)
    
    # Run
    t0 = time.time()
    # We need init params. 
    
    from dmipy_jax.fitting.initialization import GlobalBruteInitializer
    initializer = GlobalBruteInitializer(jax_mcm)
    key = jax.random.PRNGKey(42)
    candidates = initializer.generate_random_grid(2000, key)
    
    simulator_v = jax.jit(jax.vmap(jax_mcm.model_func, in_axes=(0, None)))
    cand_preds = simulator_v(candidates, jax_acq)
    
    selector_v = jax.jit(jax.vmap(initializer.select_best_candidate, in_axes=(0, None, None)))
    init_params = selector_v(data_jax, cand_preds, candidates)
    
    # Check Init Quality
    init_preds = simulator_v(init_params, jax_acq)
    mse_init = np.mean((data_jax - init_preds)**2)
    print(f"Init MSE (GPU): {mse_init:.6e}")
    
    # Run Custom LM
    fit_vmapped_custom = jax.jit(jax.vmap(custom_fitter.fit, in_axes=(0, None, 0)))
    _ = fit_vmapped_custom(data_jax[:10], jax_acq, init_params[:10]) # Warmup
    
    start_fit = time.time()
    fit_results_custom, steps_custom = fit_vmapped_custom(data_jax, jax_acq, init_params)
    fit_results_custom.block_until_ready()
    gpu_time_custom = time.time() - start_fit
    
    print(f"GPU Time (CustomLM): {gpu_time_custom:.4f} s")
    print(f"GPU Throughput (CustomLM): {N_voxels/gpu_time_custom:.2f} voxels/s")
    
    fit_results_jax = fit_results_custom 
    
    print(f"Speedup vs CPU: {cpu_time/gpu_time_custom:.2f}x")
    
    # 5. Compare
    fitted_iso_jax = fit_results_jax[:, 0]
    fitted_par_jax = fit_results_jax[:, 3]
    fitted_pv_jax = fit_results_jax[:, 4]
    
    print("\n--- Accuracy Verification ---")
    mse_cpu_pv = mean_squared_error(gt_params['partial_volume_0'], fitted_pv_cpu)
    mse_jax_pv = mean_squared_error(gt_params['partial_volume_0'], fitted_pv_jax)
    
    print(f"MSE PV0 (CPU): {mse_cpu_pv:.6e}")
    print(f"MSE PV0 (GPU): {mse_jax_pv:.6e}")
    
    r2_pv_cpu = r2_score(gt_params['partial_volume_0'], fitted_pv_cpu)
    r2_pv_jax = r2_score(gt_params['partial_volume_0'], fitted_pv_jax)
    
    print(f"R2 PV0 (CPU): {r2_pv_cpu:.4f}")
    print(f"R2 PV0 (GPU): {r2_pv_jax:.4f}")
    
    # Consistency
    r2_consistency = r2_score(fitted_pv_cpu, fitted_pv_jax)
    print(f"Consistency R2 (CPU vs GPU): {r2_consistency:.4f}")
    
    if r2_consistency > 0.95:
        print("SUCCESS: High correlation between CPU and GPU estimates.")
    else:
        print("WARNING: Low correlation. Check initialization.")
    
    # Also check Iso Diffusivity accuracy
    mse_iso_jax = mean_squared_error(gt_params['G1Ball_1_lambda_iso'], fitted_iso_jax)
    print(f"MSE Iso (GPU): {mse_iso_jax:.6e}")


if __name__ == "__main__":
    benchmark_parity()
