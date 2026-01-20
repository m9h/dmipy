
import jax
import jax.numpy as jnp
import equinox as eqx
<<<<<<< HEAD
import jaxopt
from typing import Dict, Any, Optional
=======
from scico import functional, linop, loss, optimize
from typing import List, Dict, Any, Optional
import itertools
>>>>>>> recovery_work_v2

class AMICOSolver(eqx.Module):
    """
    Accelerated Microstructure Imaging via Convex Optimization (AMICO) Solver.
    
<<<<<<< HEAD
    Uses JAXOpt ProximalGradient (FISTA) to solve:
    min_x 0.5 * || y - Phi @ x ||_2^2 + lambda * ||x||_1 + I_nonneg(x)
    
    This solver is fully JAX-transformable (jit, vmap).
    """
    
    dict_matrix: jnp.ndarray = eqx.field(init=False)
    acquisition: Any
=======
    This solver linearizes a multi-compartment model into a dictionary of atoms (kernels)
    and solves the inverse problem using convex optimization (ADMM) to find the 
    weights (volume fractions) of each atom.
    
    The problem is formulated as:
    min_x || y - Phi @ x ||_2^2 + lambda * R(x)
    s.t. x >= 0, sum(x) = 1 (optional)
    """
    
    dict_matrix: jnp.ndarray = eqx.field(init=False)
    acquisition: Any = eqx.field(static=True)
>>>>>>> recovery_work_v2
    
    def __init__(self, model: Any, acquisition: Any, dictionary_params: Dict[str, jnp.ndarray]):
        """
        Initialize AMICO Solver.
        
        Args:
<<<<<<< HEAD
            model: The microstructure model.
            acquisition: Acquisition scheme.
            dictionary_params: Dictionary to generate atoms.
=======
            model: The microstructure model (must support standard call with params).
            acquisition: Acquisition scheme.
            dictionary_params: Dictionary of parameters for the atoms. 
                               Each value should be an array of values to grid over.
                               The cartesian product of these arrays forms the dictionary atoms.
>>>>>>> recovery_work_v2
        """
        self.acquisition = acquisition
        self.dict_matrix = self.generate_kernels(model, acquisition, dictionary_params)
        
    def generate_kernels(self, model, acquisition, dictionary_params) -> jnp.ndarray:
        """
        Generates the dictionary matrix Phi [N_measurements, N_atoms].
        """
<<<<<<< HEAD
        keys = list(dictionary_params.keys())
        values = list(dictionary_params.values())
        
        # Create regular grid
        grids = jnp.meshgrid(*values, indexing='ij')
        flat_grids = [g.ravel() for g in grids]
        
        stacked_params = jnp.stack(flat_grids, axis=1)
        
=======
        # Create grid of parameters
        keys = list(dictionary_params.keys())
        values = list(dictionary_params.values())
        
        # Create regular grid (cartesian product)
        # Use meshgrid to generate all combinations
        grids = jnp.meshgrid(*values, indexing='ij')
        
        # Flatten grids to list of atoms
        flat_grids = [g.ravel() for g in grids]
        # n_atoms = len(flat_grids[0])
        
        # Evaluate model for each atom
        
        # Stack parameters: [N_atoms, N_params]
        stacked_params = jnp.stack(flat_grids, axis=1)
        
        # vmap over the stacked parameters
>>>>>>> recovery_work_v2
        def model_wrapper(p_values):
            p_dict = {k: v for k, v in zip(keys, p_values)}
            return model(p_dict, acquisition)
            
        atoms = jax.vmap(model_wrapper)(stacked_params)
<<<<<<< HEAD
=======
        
        # atoms shape: [N_atoms, N_measurements]
        # We need [N_measurements, N_atoms] for matrix multiplication y = Phi @ x
>>>>>>> recovery_work_v2
        return atoms.T

    def fit(self, data: jnp.ndarray, lambda_reg: float = 0.0, constrained: bool = True):
        """
<<<<<<< HEAD
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
=======
        Fit the model to the data using ADMM.
        
        Args:
            data: Signal data [N_measurements] or [N_voxels, N_measurements].
            lambda_reg: Regularization parameter (e.g. for L1 sparsity).
            constrained: If True, enforces non-negativity (x >= 0).
            
        Returns:
            Estimated weights [..., N_atoms]
        """
        if data.ndim == 1:
            return self._fit_single_voxel(data, lambda_reg, constrained)
        else:
            # vmap over voxels
            return jax.vmap(lambda d: self._fit_single_voxel(d, lambda_reg, constrained))(data)

    def _fit_single_voxel(self, y: jnp.ndarray, lambda_reg: float, constrained: bool):
        """
        Solves for a single voxel:
        min_x (1/2)|| y - Phi x ||_2^2 + lambda * ||x||_1 + I_nonneg(x)
        """
        # Data fidelity term: (1/2) || y - Ax ||_2^2
        dictionary_op = linop.MatrixOperator(self.dict_matrix)
        f = loss.SquaredL2Loss(y=y, A=dictionary_op)
        
        # Regularization / Constraints
        g_list = []
        C_list = []
        
        # Non-negativity constraint: I(x >= 0)
        if constrained:
            g_list.append(functional.NonNegativeIndicator())
            C_list.append(linop.Identity(self.dict_matrix.shape[1]))
            
        # L1 Regularization: lambda ||x||_1
        if lambda_reg > 0:
            g_list.append(lambda_reg * functional.L1Norm())
            C_list.append(linop.Identity(self.dict_matrix.shape[1]))
            
        if not g_list:
            # Fallback to simple least squares if no constraints/reg
            return jnp.linalg.lstsq(self.dict_matrix, y)[0]
            
        # Solve with ADMM
        solver = optimize.ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho=1.0,
            x0=jnp.zeros(self.dict_matrix.shape[1]),
            maxiter=100,
            itstat_view_func=None,
            verbose=False
        )
        
        x_hat = solver.solve()
        return x_hat

def calculate_mean_parameter_map(weights, dictionary_params, param_name):
    """
    Calculates the mean parameter map from the estimated weights.
    
    Mean param = sum(weights_i * param_i) / sum(weights_i)
    
    Args:
        weights: Estimated weights [..., N_atoms]
        dictionary_params: Dictionary of parameter grids used to generate atoms.
        param_name: Name of the parameter to compute mean for.
        
    Returns:
        Mean parameter map [...,]
    """
    keys = list(dictionary_params.keys())
    values = list(dictionary_params.values())
    
    # Generate all combinations exactly as in generate_kernels
    grids = jnp.meshgrid(*values, indexing='ij')
    flat_grids = [g.ravel() for g in grids]
    
    # Find index of param_name
    try:
        idx = keys.index(param_name)
    except ValueError:
        raise ValueError(f"Parameter {param_name} not found in dictionary parameters: {keys}")
        
    param_values_per_atom = flat_grids[idx] # (N_atoms,)
    
    # Compute weighted average
    # weights: (..., N_atoms)
    # param_values: (N_atoms,)
    
    # jnp.dot uses last dim of weights if weights is ND?
    # Actually weighted_sum = sum(w * p, axis=-1)
    
    # We must be careful with broadcasting.
    # weights: [..., N]
    # param_values: [N]
    
    weighted_sum = jnp.dot(weights, param_values_per_atom)
    
    # normalization: sum(weights)
    normalization = jnp.sum(weights, axis=-1)
    
    # Avoid division by zero
    normalization = jnp.where(normalization == 0, 1.0, normalization)
    
>>>>>>> recovery_work_v2
    return weighted_sum / normalization
