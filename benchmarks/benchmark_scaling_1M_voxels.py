import time
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
import dmipy_jax.signal_models.gaussian_models as jax_gaussian
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def benchmark_scaling():
    print("Setting up 1M voxel benchmark...")
    
    # ----------------------------------------------------------------
    # 2. Setup DMIPY-JAX Models (Before Data for Simulation)
    # ----------------------------------------------------------------
    ball_jax = jax_gaussian.Ball() # Use correct class name
    stick_jax = Stick()            # Use correct class name
    
    # Set ranges on the instances directly (using their internal names)
    ball_jax.parameter_ranges['lambda_iso'] = (1e-9, 4e-9)
    stick_jax.parameter_ranges['lambda_par'] = (1e-9, 3e-9)
    # mu is default full sphere
    
    jax_mcm = JaxMultiCompartmentModel(models=[ball_jax, stick_jax])

    # ----------------------------------------------------------------
    # 1. Setup Data (1 Million Voxels)
    # ----------------------------------------------------------------
    N_voxels = 1_000_000
    
    # Use standard HCP scheme
    acq_scheme_dmipy = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme_jax = JaxAcquisition(
        bvalues=acq_scheme_dmipy.bvalues,
        gradient_directions=acq_scheme_dmipy.gradient_directions,
        delta=acq_scheme_dmipy.delta,
        Delta=acq_scheme_dmipy.Delta
    )
    
    # Define Model: Ball + Stick (Legacy for GT if needed, but we use JAX now)
    # mc_model = MultiCompartmentModel(models=[...]) # Not used for simulation anymore
    
    print(f"Generating data for {N_voxels} voxels...")
    np.random.seed(42)
    
    gt_params = {}
    gt_params['partial_volume_0'] = np.random.uniform(0.1, 0.9, N_voxels)
    gt_params['partial_volume_1'] = 1 - gt_params['partial_volume_0']
    gt_params['G1Ball_1_lambda_iso'] = np.full(N_voxels, 3e-9)
    gt_params['C1Stick_1_lambda_par'] = np.full(N_voxels, 1.7e-9)
    theta = np.random.uniform(0, np.pi, N_voxels)
    phi = np.random.uniform(0, 2*np.pi, N_voxels)
    gt_params['C1Stick_1_mu'] = np.column_stack([theta, phi])
    
    # Template params
    N_template = 10_000
    gt_params_small = {k: v[:N_template] if len(v) == N_voxels else v for k, v in gt_params.items()}
    gt_params_small['C1Stick_1_mu'] = gt_params['C1Stick_1_mu'][:N_template]

    # Simulate signal (using JAX model)
    sim_params = {}
    sim_params['partial_volume_0'] = gt_params_small['partial_volume_0']
    sim_params['partial_volume_1'] = gt_params_small['partial_volume_1']
    sim_params['lambda_iso'] = gt_params_small['G1Ball_1_lambda_iso']
    sim_params['lambda_par'] = gt_params_small['C1Stick_1_lambda_par']
    sim_params['mu'] = gt_params_small['C1Stick_1_mu']
    
    print("Simulating template signal (JAX)...")
    data_small = jax_mcm(sim_params, acq_scheme_jax)
    data_small = np.array(data_small)
    
    # Tile to 1M
    tile_count = N_voxels // N_template
    data_large = np.tile(data_small, (tile_count, 1))
    
    print(f"Data shape: {data_large.shape} ({data_large.nbytes / 1e9:.2f} GB)")
    
    # Transfer to JAX
    print("Moving data to GPU...")
    data_jax = jnp.array(data_large)
    
    # ----------------------------------------------------------------
    # 3. Benchmark
    # ----------------------------------------------------------------
    
    # Batch size for chunking
    # Initialization step is memory intensive, so we use smaller batch size
    BATCH_SIZE = 5000
    
    print("\n--- Compiling (Warmup) ---")
    
    # We wrap the fit call
    # Note: JIT-ing the outer loop over chunks unrolls it. 
    # For 5 chunks it is fine.
    
    # We wrap the fit call
    # Note: We do NOT jit here, so the loop inside fit runs in Python
    # dispensing chunks to the JIT-ed kernel.
    
    def run_fit(data_batch):
        return jax_mcm.fit(acq_scheme_jax, data_batch, compute_uncertainty=False, batch_size=BATCH_SIZE)
        
    # Warmup with one chunk to compile the inner kernel
    print(f"Compiling kernel for batch size {BATCH_SIZE}...")
    warmup_data = data_jax[:BATCH_SIZE]
    _ = run_fit(warmup_data)
    jax.block_until_ready(warmup_data) # wait for compilation
    print("Compilation complete.")
    
    print(f"\n--- Running 1M Voxel Fit (Chunked) ---")
    
    start_time = time.time()
    
    # Run fit on FULL data (fit methods handles chunking)
    res = run_fit(data_jax)
    
    # Block to ensure completion
    first_key = list(res.keys())[0]
    res[first_key].block_until_ready()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total Time: {total_time:.4f} s")
    print(f"Time per Voxel: {total_time / N_voxels * 1e6:.2f} us")
    
    if total_time < 5.0:
        print("\n[SUCCESS] Benchmark goal met (< 5s)!")
    else:
        print("\n[FAIL] Benchmark goal missed (> 5s).")

if __name__ == "__main__":
    benchmark_scaling()
