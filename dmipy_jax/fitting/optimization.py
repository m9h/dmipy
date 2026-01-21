import jax
import jax.numpy as jnp
import jaxopt
import optimistix as optx
from functools import partial

__all__ = ['VoxelFitter', 'OptimistixFitter', 'CustomLMFitter', 'mse_loss']

def mse_loss(params, data, bvals, bvecs, model_func, unwrap_fn):
    args = unwrap_fn(params)
    prediction = model_func(bvals, bvecs, *args)
    return jnp.mean((data - prediction) ** 2)

class VoxelFitter:
    """
    A JAX-optimized fitter using the Pure-JAX L-BFGS-B solver.
    This allows full compilation and vectorization on the GPU.
    """
    def __init__(self, model_func, parameter_ranges, solver_settings=None, scales=None):
        self.model_func = model_func
        
        # Scales for parameters (default 1.0)
        # p_internal = p_real / scales
        # p_real = p_internal * scales
        n_params = len(parameter_ranges)
        if scales is None:
            self.scales = jnp.ones(n_params)
        else:
            self.scales = jnp.array(scales)
            
        # Convert list of (min, max) into tuple of arrays for JAX
        # Bounds must be scaled for the solver
        lower_bounds_real, upper_bounds_real = zip(*parameter_ranges)
        self.bounds = (
            jnp.array(lower_bounds_real) / self.scales, 
            jnp.array(upper_bounds_real) / self.scales
        )
        
        # Default settings
        self.solver_settings = {
            'maxiter': 100,
            'tol': 1e-5
        }
        if solver_settings:
            self.solver_settings.update(solver_settings)

    # Static argnums tells JAX to recompile if 'self' changes (which it won't here)
    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data, acquisition, init_params):
        """
        Fits a single voxel. Can be vmapped over 'data' and 'init_params'.
        
        Args:
            data: (N_measurements,) array of signal data.
            acquisition: JaxAcquisition object.
            init_params: (N_params,) array of initial guess (REAL scale).
        """
        
        # Scale initial params for the solver
        init_params_solver = init_params / self.scales
        
        # Define objective closure (operating on scaled params)
        def objective(params_solver, data, acquisition):
            # Descale to get real parameters for the model
            params_real = params_solver * self.scales
            
            # model_func signature: (params, acquisition) -> signal
            prediction = self.model_func(params_real, acquisition)
            # MSE Loss
            return jnp.mean((data - prediction) ** 2)

        # Instantiate Pure JAX Solver (L-BFGS-B)
        solver = jaxopt.LBFGSB(
            fun=objective,
            **self.solver_settings
        )

        # Run Solver
        sol = solver.run(
            init_params_solver, 
            bounds=self.bounds,
            data=data,
            acquisition=acquisition
        )
        
        # Return fitted params in REAL scale
        fitted_params_real = sol.params * self.scales
        return fitted_params_real, sol.state

class OptimistixFitter:
    """
    A JAX-optimized fitter using Optimistix's Levenberg-Marquardt solver.
    This provides 2nd-order convergence and robust handling of non-linear squares.
    """
    def __init__(self, model_func, parameter_ranges, solver_settings=None, scales=None):
        self.model_func = model_func
        
        n_params = len(parameter_ranges)
        if scales is None:
            self.scales = jnp.ones(n_params)
        else:
            self.scales = jnp.array(scales)
            
        # Optimistix handles bounds differently? 
        # Actually optimistix.least_squares doesn't natively support box constraints 
        # in the same way L-BFGS-B does easily without extra work (e.g. projecting).
        # However, for now we will implement unconstrained LM or check if we can add constraints.
        # Levenberg-Marquardt is typically unconstrained.
        # We can simulate constraints by identifying parameter transforms (sigmoid/softplus) outside.
        # BUT existing VoxelFitter exposes bounds.
        # WARNING: optimistix.least_squares does NOT support bounds directly in the solver args like jaxopt.
        # We might need to stick to unconstrained for the first pass or enforce it via parameter mapping.
        # For this implementation, we will assume UNCONSTRAINED optimization for LM 
        # (or rely on the user to map params).
        # But wait, roadmap says "NaN-safe Trust Regions".
        
        self.solver_settings = {
            'rtol': 1e-5,
            'atol': 1e-5
        }
        if solver_settings:
            self.solver_settings.update(solver_settings)

    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data, acquisition, init_params):
        """
        Fits a single voxel using Levenberg-Marquardt.
        """
        # Capture self attributes locally to verify if 'self' capture is causing Equinox issues
        scales = self.scales
        model_func = self.model_func

        # Scale initial params
        init_params_solver = init_params / scales

        # Residual function for least_squares
        # Returns vector of residuals: predictions - data
        # We capture acquisition and data from the outer scope to avoid passing them as args
        # which seems to trigger Equinox tracing issues with PyTree tracers.
        def residual_fun(params_solver, _):
            # acquisition and data are captured from the outer scope
            params_real = params_solver * scales
            prediction = model_func(params_real, acquisition)
            return prediction - data

        # Solver
        solver = optx.LevenbergMarquardt(
            rtol=self.solver_settings['rtol'], 
            atol=self.solver_settings['atol']
        )
        
        # Run least_squares
        sol = optx.least_squares(
            fn=residual_fun,
            solver=solver,
            y0=init_params_solver, # Initial guess
            args=(), # API requires args, passing empty tuple
            max_steps=100, # Safety break
            throw=False # Don't crash on non-convergence
        )

        fitted_params_real = sol.value * self.scales
        # Return params and step count
        # Handle case where stats might be missing (though unlikely with LM)
        steps = sol.stats.get('num_steps', jnp.array(-1))
        return fitted_params_real, steps

from dmipy_jax.fitting.custom_solvers import BatchedLevenbergMarquardt

class CustomLMFitter:
    """
    A High-Performance Fitter using a custom Batched Levenberg-Marquardt solver.
    Designed for massive throughput on GPU by using unrolled loops and Cholesky factorization.
    """
    def __init__(self, model_func, parameter_ranges, solver_settings=None, scales=None):
        self.model_func = model_func
        
        n_params = len(parameter_ranges)
        if scales is None:
            self.scales = jnp.ones(n_params)
        else:
            self.scales = jnp.array(scales)
            
        self.solver_settings = {
            'max_steps': 20,
            'damping': 1e-3
        }
        if solver_settings:
            self.solver_settings.update(solver_settings)
            
    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data, acquisition, init_params):
        """
        Fits a single voxel using Custom Batched LM.
        """
        scales = self.scales
        model_func = self.model_func

        init_params_solver = init_params / scales

        # Residual function
        def residual_fun(params_solver, _):
            params_real = params_solver * scales
            prediction = model_func(params_real, acquisition)
            return prediction - data

        solver = BatchedLevenbergMarquardt(
            max_steps=self.solver_settings['max_steps'],
            damping=self.solver_settings['damping']
        )
        
        final_params_solver, stats = solver.solve(residual_fun, init_params_solver)
        
        fitted_params_real = final_params_solver * scales
        steps = stats['steps'] # Fixed steps
        
        return fitted_params_real, steps
