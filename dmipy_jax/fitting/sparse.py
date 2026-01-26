import jax
import jax.numpy as jnp
from functools import partial
from dmipy_jax.fitting.custom_solvers import BatchedProximalLevenbergMarquardt
from dmipy_jax.utils.proximal import non_negative_soft_thresholding

class SparsityLMFitter:
    """
    A Fitter that enforces L1 sparsity on specific parameters (e.g. Volume Fractions)
    using a Proximal Levenberg-Marquardt solver.
    """
    
    def __init__(self, model_func, parameter_ranges, sparsity_lambda=0.1, n_fractions=0, scales=None):
        self.model_func = model_func
        self.sparsity_lambda = sparsity_lambda
        self.n_fractions = n_fractions # Number of fraction parameters at the END of params
        
        n_params = len(parameter_ranges)
        if scales is None:
            self.scales = jnp.ones(n_params)
        else:
            self.scales = jnp.array(scales)
            
        self.solver_settings = {
            'max_steps': 50,
            'damping': 1e-3
        }
            
    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data, acquisition, init_params):
        """
        Fits a single voxel using Proximal LM.
        """
        scales = self.scales
        model_func = self.model_func
        n_params = len(scales)
        n_fractions = self.n_fractions
        lambda_val = self.sparsity_lambda

        init_params_solver = init_params / scales

        # 1. Define Residual Function
        def residual_fun(params_solver, _):
            params_real = params_solver * scales
            prediction = model_func(params_real, acquisition)
            return prediction - data

        # 2. Define Proximal Operator
        # Applies to solver_scale params?
        # Ideally we apply soft threshold to the real fractions.
        # But the solver operates on scaled params.
        # prox_solver(p) = prox_real(p * s) / s
        
        # Soft Thresholding on Fractions only.
        # Fractions are the last n_fractions parameters.
        # We assume fractions are scaled by 1.0 usually.
        
        def prox_op(params_solver):
            # Split params
            k_structural = n_params - n_fractions
            
            p_struct = params_solver[:k_structural]
            p_fracs = params_solver[k_structural:]
            
            # Apply Soft Thresholding to Fractions
            # Note: We incorporate non-negativity constraint here too
            # proximal = max(0, x - lambda)
            p_fracs_new = non_negative_soft_thresholding(p_fracs, lambda_val)
            
            return jnp.concatenate([p_struct, p_fracs_new])

        # 3. Solve
        solver = BatchedProximalLevenbergMarquardt(
            max_steps=self.solver_settings['max_steps'],
            damping=self.solver_settings['damping']
        )
        
        final_params_solver, stats = solver.solve(residual_fun, init_params_solver, prox_op)
        
        fitted_params_real = final_params_solver * scales
        steps = stats['steps']
        
        return fitted_params_real, steps
