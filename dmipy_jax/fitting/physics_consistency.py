import jax
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.custom_solvers import BatchedLevenbergMarquardt

def physics_loss(sr_signal, bvals, bvecs, model, init_params=None, solver_settings=None):
    """
    Computes a physics consistency loss by fitting a biophysical model to the super-resolved signal
    and measuring the reconstruction error (distance to the physical manifold).

    This function is fully differentiable (via unrolled optimization), allowing gradients to flow
    from the reconstruction error back to the `sr_signal` input.

    Args:
        sr_signal: (N_bvals,) array containing the super-resolved DWI signal for a single voxel.
                   (Batching should be handled via vmap by the caller).
        bvals: (N_bvals,) array of b-values.
        bvecs: (N_bvals, 3) array of gradient directions.
        model: A dmipy-jax model instance (e.g. Stick, BallStick).
               Must have `parameter_ranges` and `__call__` method.
        init_params: (Optional) Initial guess for the parameters. If None, uses the mean of parameter ranges.
        solver_settings: (Optional) Dict of settings for BatchedLevenbergMarquardt (e.g., 'max_steps', 'damping').

    Returns:
        Scalar MSE between `sr_signal` and the best-fit model reconstruction.
    """
    
    # 1. Setup Acquisition
    # Wrapper for bvals/bvecs as expected by many dmipy-jax components
    acquisition = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    # 2. Determine Parameter Ranges and Scaling
    if hasattr(model, 'parameter_ranges'):
        ranges = model.parameter_ranges
        # Ensure we have parameter names and cardinality
        if hasattr(model, 'parameter_names'):
            param_names = model.parameter_names
        else:
            param_names = sorted(ranges.keys())
            
        if hasattr(model, 'parameter_cardinality'):
            cardinality = model.parameter_cardinality
        else:
            # Infer cardinality
            cardinality = {}
            for name in param_names:
                r = ranges[name]
                # Check if first element is scalar (float/int/array-scalar)
                if isinstance(r[0], (int, float)) or (hasattr(r[0], 'ndim') and r[0].ndim == 0):
                    cardinality[name] = 1
                else:
                    cardinality[name] = len(r)

        lower_bounds = []
        upper_bounds = []
        
        # Helper to process a single range tuple (min, max)
        def add_range(r):
            lower_bounds.append(r[0])
            upper_bounds.append(r[1])

        for name in param_names:
            r = ranges[name]
            k = cardinality.get(name, 1)
            
            if k == 1:
                # Expect (min, max)
                add_range(r)
            else:
                # Expect sequence of (min, max)
                # e.g. ([0, pi], [-pi, pi])
                if len(r) != k:
                     raise ValueError(f"Cardinality mismatch for {name}: expected {k}, got {len(r)} ranges.")
                for sub_r in r:
                    add_range(sub_r)
            
        lower_bounds = jnp.array(lower_bounds)
        upper_bounds = jnp.array(upper_bounds)
        
        # Scaling strategy: 
        # Use simple heuristic based on magnitude of bounds
        # Scale = max(abs(lb), abs(ub))? 
        # Or just checking name.
        scales = []
        idx = 0
        for name in param_names:
            k = cardinality.get(name, 1)
            for _ in range(k):
                # Check name for diffusivity
                if 'diffusivity' in name or 'lambda' in name: 
                    scales.append(1e-9)
                else:
                    scales.append(1.0)
                idx += 1
        scales = jnp.array(scales)

    else:
        raise ValueError("Model must have `parameter_ranges` parameter.")

    # 3. Initial Guess
    if init_params is None:
        init_params_real = (lower_bounds + upper_bounds) / 2.0
    else:
        init_params_real = init_params
        
    init_params_solver = init_params_real / scales

    # 4. Define Residual Function for Solver
    def residual_fun(params_solver, _):
        params_real = params_solver * scales
        
        kwargs = {}
        curr_idx = 0
        for name in param_names:
            k = cardinality.get(name, 1)
            if k == 1:
                kwargs[name] = params_real[curr_idx]
                curr_idx += 1
            else:
                kwargs[name] = params_real[curr_idx : curr_idx + k]
                curr_idx += k
             
        prediction = model(bvals=bvals, gradient_directions=bvecs, **kwargs)
        
        return prediction - sr_signal

    # 5. Run Solver
    default_settings = {'max_steps': 10, 'damping': 1e-3}
    if solver_settings:
        default_settings.update(solver_settings)
        
    solver = BatchedLevenbergMarquardt(
        max_steps=default_settings['max_steps'],
        damping=default_settings['damping']
    )
    
    # We don't have extra args for residual_fun, everything captured in closure 
    # (which is fine for JAX as long as they are array tracers)
    final_params_solver, _ = solver.solve(residual_fun, init_params_solver)
    
    # 6. Compute Final Loss (Reconstruction Error)
    # This is the "Physics Loss"
    final_residuals = residual_fun(final_params_solver, None)
    mse = jnp.mean(final_residuals**2)
    
    return mse
