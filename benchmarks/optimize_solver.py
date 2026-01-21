
import time
import jax
import jax.numpy as jnp
import optimistix as optx
import lineax as lx
import numpy as np
from dmipy_jax.signal_models import gaussian_models, stick
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition
from dmipy.data import saved_acquisition_schemes

def benchmark_solver_config():
    print("Setting up solver benchmark...")
    
    # 1. Setup Models & Data (Small scale: 5000 voxels)
    ball = gaussian_models.Ball()
    stick_model = stick.Stick()
    ball.parameter_ranges['lambda_iso'] = (1e-9, 4e-9)
    stick_model.parameter_ranges['lambda_par'] = (1e-9, 3e-9)
    jax_mcm = JaxMultiCompartmentModel(models=[ball, stick_model])
    
    N_voxels = 5000
    acq_scheme_dmipy = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_scheme_jax = JaxAcquisition(
        bvalues=acq_scheme_dmipy.bvalues,
        gradient_directions=acq_scheme_dmipy.gradient_directions,
        delta=acq_scheme_dmipy.delta,
        Delta=acq_scheme_dmipy.Delta
    )
    
    # Generate Data
    key = jax.random.PRNGKey(42)
    gt_params = {
        'partial_volume_0': jax.random.uniform(key, (N_voxels,), minval=0.1, maxval=0.9),
        'lambda_iso': jnp.full((N_voxels,), 3e-9),
        'lambda_par': jnp.full((N_voxels,), 1.7e-9),
        'mu': jnp.column_stack([
             jax.random.uniform(key, (N_voxels,), minval=0, maxval=np.pi),
             jax.random.uniform(key, (N_voxels,), minval=0, maxval=2*np.pi)
        ])
    }
    gt_params['partial_volume_1'] = 1 - gt_params['partial_volume_0']
    
    # Helper to flatten
    gt_list = []
    for name in jax_mcm.parameter_names:
        val = gt_params[name]
        if jax_mcm.parameter_cardinality[name] == 1:
            gt_list.append(val[:, None])
        else:
            gt_list.append(val)
    gt_array = jnp.hstack(gt_list)
    
    print("Simulating...")
    data_jax = jax.vmap(jax_mcm.model_func, in_axes=(0, None))(gt_array, acq_scheme_jax)
    data_jax = jax.device_put(data_jax)
    
    # Init guess (perturbed GT)
    init_params = gt_array * 1.1
    
    # 2. Define Fit Function Factory
    def make_fit_fn(linear_solver_method):
        
        scales = jnp.ones(6) # Simplified scales
        
        def residual_fun(params_solver, _):
            params_real = params_solver * scales
            prediction = jax_mcm.model_func(params_real, acq_scheme_jax)
            return prediction - data_jax[0] # Dummy for shape logic? No, this is vmap inner.
            
        # We need a vmappable fit function
        def fit_single(data_voxel, init_p):
            def res_fn(p, _):
                p_real = p * scales
                pred = jax_mcm.model_func(p_real, acq_scheme_jax)
                return pred - data_voxel
                
            solver = optx.LevenbergMarquardt(
                rtol=1e-5, atol=1e-5,
                linear_solver=linear_solver_method
            )
            sol = optx.least_squares(
                fn=res_fn, solver=solver, y0=init_p/scales, args=(), 
                max_steps=20, throw=False
            )
            return sol.value, sol.stats['num_steps']
            
        return jax.jit(jax.vmap(fit_single))
    # 3. Benchmark Configurations
    configs = [
        ("Auto", lx.AutoLinearSolver(well_posed=False)), 
        ("Cholesky", lx.Cholesky()), 
        ("QR", lx.QR()),
        ("SVD", lx.SVD())
    ]
    
    for name, ls in configs:
        print(f"\n--- Testing Linear Solver: {name} ---")
        try:
            fit_fn = make_fit_fn(ls)
            
            # Warmup
            print("Compiling...")
            _ = fit_fn(data_jax[:10], init_params[:10])
            
            # Run
            print("Running...")
            jax.block_until_ready(data_jax)
            t0 = time.time()
            res, steps = fit_fn(data_jax, init_params)
            res.block_until_ready()
            dt = time.time() - t0
            
            print(f"Time (5000 voxels): {dt:.4f} s")
            print(f"Throughput: {N_voxels/dt:.2f} voxels/s")
            print(f"Avg Steps: {jnp.mean(steps):.2f}")
            
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    benchmark_solver_config()
