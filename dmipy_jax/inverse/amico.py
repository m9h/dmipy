
import jax
import jax.numpy as jnp
import equinox as eqx
import jaxopt
from typing import Dict, Any, Optional

class AMICOSolver(eqx.Module):
    """
    Accelerated Microstructure Imaging via Convex Optimization (AMICO) Solver.
    
    Uses JAXOpt ProximalGradient (FISTA) to solve:
    min_x 0.5 * || y - Phi @ x ||_2^2 + lambda * ||x||_1 + I_nonneg(x)
    
    This solver is fully JAX-transformable (jit, vmap).
    """
    
    dict_matrix: jnp.ndarray = eqx.field(init=False)
    acquisition: Any
    
    def __init__(self, model: Any, acquisition: Any, dictionary_params: Dict[str, jnp.ndarray]):
        """
        Initialize AMICO Solver.
        
        Args:
            model: The microstructure model.
            acquisition: Acquisition scheme.
            dictionary_params: Dictionary to generate atoms.
        """
        self.acquisition = acquisition
        self.dict_matrix = self.generate_kernels(model, acquisition, dictionary_params)
        
    def generate_kernels(self, model, acquisition, dictionary_params) -> jnp.ndarray:
        """
        Generates the dictionary matrix Phi [N_measurements, N_atoms].
        """
        keys = list(dictionary_params.keys())
        values = list(dictionary_params.values())
        
        # Create regular grid
        grids = jnp.meshgrid(*values, indexing='ij')
        flat_grids = [g.ravel() for g in grids]
        
        stacked_params = jnp.stack(flat_grids, axis=1)
        
        def model_wrapper(p_values):
            p_dict = {k: v for k, v in zip(keys, p_values)}
            return model(p_dict, acquisition)
            
        atoms = jax.vmap(model_wrapper)(stacked_params)
        return atoms.T

    def fit(self, data: jnp.ndarray, lambda_reg: float = 0.0, constrained: bool = True):
        """
        Fit the model using Proximal Gradient Descent (FISTA).
        
        Args:
            data: [N_meas] or [..., N_meas] signal.
            lambda_reg: L1 regularization strength.
            constrained: Enforce non-negativity.
        """
        
        # Define objective: 0.5 * || y - A x ||^2
        # Note: dict_matrix is fixed (self).
        def objective_fun(x, y):
            residuals = y - self.dict_matrix @ x
            return 0.5 * jnp.sum(residuals**2)
            
        # Define Prox
        # Prox for g(x) = lambda ||x||_1 + I_nonneg(x)
        # prox(v) = argmin_x 0.5 ||x - v||^2 + g(x)
        # Solution: ReLU(SoftThreshold(v, lambda)) = ReLU(v - lambda) (since lambda>0, max(v-lambda, 0))
        # Taking into account 'scaling' from step-size in ProximalGradient.
        def prox_fun(params, hyperparams_prox=None, scaling=1.0):
            lam = hyperparams_prox
            threshold = lam * scaling
            
            if constrained:
                # L1 + NonNeg
                # x = max(params - threshold, 0)
                return jax.nn.relu(params - threshold)
            else:
                # Just L1
                # x = sign(params) * max(|params| - threshold, 0)
                return jaxopt.prox.prox_l1(params, threshold) # jaxopt prox_l1 takes threshold directly as 2nd arg? 
                                                              # No, signature is (x, modulus, scaling).
                                                              # jaxopt.prox.prox_l1(x, lam, scaling) -> soft_threshold(x, lam*scaling)
                # Ensure we use it correctly. checking docs: 
                # prox_l1(params, hyperparams_prox=None, scaling=1.0)
                # Returns soft_threshold(params, hyperparams_prox * scaling)
                return jaxopt.prox.prox_l1(params, lam, scaling)

        # Initialize solver
        # We assume explicit use of self.dict_matrix inside objective_fun is fin w.r.t Equinox modules.
        # Yes, self.dict_matrix is a JAX array in the module.
        
        solver = jaxopt.ProximalGradient(
            fun=objective_fun,
            prox=prox_fun,
            maxiter=500,
            tol=1e-5,
            acceleration=True # FISTA
        )
        
        n_atoms = self.dict_matrix.shape[1]
        x0 = jnp.zeros(n_atoms)
        
        # Run
        if data.ndim == 1:
            sol = solver.run(x0, hyperparams_prox=lambda_reg, y=data)
            return sol.params
        else:
            # vmap over data
            def run_single(y):
                # We pass 'y' as keyword arg if it matches objective_fun signature parameter name?
                # jaxopt passes *args and **kwargs to fun. 
                # objective_fun(x, y). So 'y' can be pos or kw.
                sol = solver.run(x0, hyperparams_prox=lambda_reg, y=y)
                return sol.params
            
            return jax.vmap(run_single)(data)

def calculate_mean_parameter_map(weights, dictionary_params, param_name):
    """Calculates the mean parameter map from the estimated weights."""
    keys = list(dictionary_params.keys())
    values = list(dictionary_params.values())
    grids = jnp.meshgrid(*values, indexing='ij')
    flat_grids = [g.ravel() for g in grids]
    
    try:
        idx = keys.index(param_name)
    except ValueError:
        raise ValueError(f"Parameter {param_name} not found.")
        
    param_values_per_atom = flat_grids[idx]
    
    weighted_sum = jnp.dot(weights, param_values_per_atom)
    normalization = jnp.sum(weights, axis=-1)
    normalization = jnp.where(normalization == 0, 1.0, normalization)
    return weighted_sum / normalization
