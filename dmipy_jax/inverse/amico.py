
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import unxt
import scico
from scico import functional, linop, loss, optimize
from typing import List, Dict, Any, Optional
from jaxtyping import Float, Array, Int, Bool

class AMICOSolver(eqx.Module):
    """
    Accelerated Microstructure Imaging via Convex Optimization (AMICO) Solver.
    
    This solver linearizes a multi-compartment model into a dictionary of atoms (kernels)
    and solves the inverse problem using convex optimization (ADMM) to find the 
    weights (volume fractions) of each atom.
    
    The problem is formulated as:
    min_x || y - Phi @ x ||_2^2 + lambda * R(x)
    s.t. x >= 0, sum(x) = 1 (optional)
    """
    
    dict_operator: linop.MatrixOperator
    acquisition: Any
    dict_unit: Optional[Any] = None
    
    @classmethod
    def create(cls, model: Any, acquisition: Any, dictionary_params: Dict[str, Float[Array, "..."]]):
        """
        Creates an AMICOSolver instance.
        
        Args:
            model: The microstructure model.
            acquisition: Acquisition scheme.
            dictionary_params: Dictionary of parameters for the atoms. 
        """
        dict_matrix = cls.generate_kernels(model, acquisition, dictionary_params)
        
        # Unit Handling
        dict_unit = None
        if hasattr(dict_matrix, 'unit'):
            dict_unit = dict_matrix.unit
            dict_matrix = dict_matrix.ustrip(dict_unit)

        dict_operator = linop.MatrixOperator(dict_matrix)

        return cls(dict_operator=dict_operator, acquisition=acquisition, dict_unit=dict_unit)

    @staticmethod
    def generate_kernels(model, acquisition, dictionary_params) -> Float[Array, "measurements atoms"]:
        """
        Generates the dictionary matrix Phi [N_measurements, N_atoms].
        """
        # Create grid of parameters
        keys = list(dictionary_params.keys())
        values = list(dictionary_params.values())
        
        # Handle units for meshgrid (strip for grid generation, re-attach for model)
        stripped_values = []
        units = []
        for v in values:
            if hasattr(v, 'unit'):
                units.append(v.unit)
                # Use ustrip to get magnitude
                stripped_values.append(v.ustrip(v.unit))
            else:
                units.append(None)
                stripped_values.append(v)

        # Create regular grid (cartesian product)
        grids = jnp.meshgrid(*stripped_values, indexing='ij')
        
        # Flatten grids to list of atoms
        flat_grids = [g.ravel() for g in grids]
        n_atoms = len(flat_grids[0])
        
        # Evaluate model for each atom
        def model_wrapper(p_values):
            p_dict = {}
            for i, k in enumerate(keys):
                val = p_values[i]
                u = units[i]
                if u is not None:
                    p_dict[k] = unxt.Quantity(val, u)
                else:
                    p_dict[k] = val
            return model(p_dict, acquisition)

        # Stack parameters: [N_atoms, N_params]
        stacked_params = jnp.stack(flat_grids, axis=1)
        atoms = jax.vmap(model_wrapper)(stacked_params)
        
        # atoms shape: [N_atoms, N_measurements]
        # We need [N_measurements, N_atoms] for matrix multiplication y = Phi @ x
        return atoms.T

    @property
    def dict_matrix(self) -> Float[Array, "measurements atoms"]:
        # Backward compatibility / Access to underlying matrix
        return self.dict_operator.A

    def _fit_batch(self, Y_batch: Float[Array, "measurements batch"], LHS: Float[Array, "atoms atoms"], c_and_lower: tuple, lambda_reg: float, rho: float, constrained: bool, max_iter: int) -> Float[Array, "atoms batch"]:
        """
        Internal batch fitting function (to be JIT-compiled).
        """
        # Access underlying matrix for Cholesky solve
        # (This is allowed as we are inside the solver implementation)
        A = self.dict_operator.A
        K = A.shape[1]

        AtY = A.T @ Y_batch # [N_atoms, Batch]
        
        x = jnp.zeros((K, Y_batch.shape[1]))
        z = jnp.zeros_like(x)
        u = jnp.zeros_like(x)
        
        # Scico Functionals
        g_nonneg = functional.NonNegativeIndicator()
        g_l1 = functional.L1Norm()

        def admm_step(carry, _):
            x, z, u = carry
            rhs = AtY + rho * (z - u)
            x_new = jax.scipy.linalg.cho_solve(c_and_lower, rhs)
            v = x_new + u
            
            # Proximal Steps
            if constrained:
                v_proj = g_nonneg.prox(v, 1.0)
            else:
                v_proj = v
            
            if lambda_reg > 0:
                 # L1 Prox: prox_{step * ||.||_1}
                 # We want threshold = lambda_reg / rho.
                 # So step = lambda_reg / rho.
                 z_new = g_l1.prox(v_proj, lambda_reg / rho)
            else:
                 z_new = v_proj
                 
            u_new = u + x_new - z_new
            return (x_new, z_new, u_new), None

        final_carry, _ = jax.lax.scan(admm_step, (x, z, u), None, length=max_iter)
        X_hat, _, _ = final_carry
        return X_hat

    def fit(self, data: Float[Array, "... measurements"], lambda_reg: float = 0.0, constrained: bool = True, batch_size: int = 10000, max_iter: int = 1000, rho: float = 1.0) -> Float[Array, "... atoms"]:
        """
        Fit the model to the data using ADMM.
        """
        # Unit Sandwich
        was_quantity = False
        if hasattr(data, 'unit'):
            was_quantity = True
            if self.dict_unit is not None:
                # Ensure units match dictionary
                data = data.uconvert(self.dict_unit)
                data_mag = data.ustrip(self.dict_unit)
            else:
                # Dictionary has no units, strip data units (assume match)
                data_mag = data.ustrip(data.unit)
        else:
            data_mag = data

        # Handle single voxel case
        if data_mag.ndim == 1:
            data_mag = data_mag[None, :]
            was_1d = True
        else:
            was_1d = False

        # Input data is [N_voxels, N_measurements]
        N_voxels = data_mag.shape[0]
        N_atoms = self.dict_matrix.shape[1]
        
        # Precompute Solver Matrices (Shared across batches)
        needs_admm = constrained or (lambda_reg > 0)
        
        if needs_admm:
            # rho is arg
            A = self.dict_matrix
            AtA = A.T @ A
            LHS = AtA + rho * jnp.eye(N_atoms)
            c_and_lower = jax.scipy.linalg.cho_factor(LHS)
            
            @jax.jit
            def run_batch(d_batch):
                 return self._fit_batch(d_batch.T, LHS, c_and_lower, lambda_reg, rho, constrained, max_iter)

            results = []
            num_batches = int(np.ceil(N_voxels / batch_size))
            
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, N_voxels)
                batch_data = data_mag[start:end]
                batch_res = run_batch(batch_data)
                results.append(batch_res.T)
                
            X_hat = jnp.concatenate(results, axis=0)

        else:
             @jax.jit
             def run_lstsq(d_batch):
                  return jnp.linalg.lstsq(self.dict_matrix, d_batch.T)[0].T

             results = []
             num_batches = int(np.ceil(N_voxels / batch_size))
             for i in range(num_batches):
                 start = i * batch_size
                 end = min((i + 1) * batch_size, N_voxels)
                 results.append(run_lstsq(data[start:end]))
                 
             X_hat = jnp.concatenate(results, axis=0)
        
        if was_1d:
            result = X_hat[0]
        else:
            result = X_hat

        if was_quantity:
            # Re-attach dimensionless unit for weights
            # "dimensionless" string might fail in some envs, using unit division safety
            u_dim = unxt.unit("m") / unxt.unit("m")
            return unxt.Quantity(result, u_dim)
        return result

def calculate_mean_parameter_map(weights: Float[Array, "... atoms"], dictionary_params: Dict[str, Float[Array, "..."]], parameter_name: str) -> Float[Array, "..."]:
    """
    Calculates the mean parameter map from the estimated weights.
    
    Mean = Sum(w_i * p_i)
    """
    keys = list(dictionary_params.keys())
    values = list(dictionary_params.values())
    
    if parameter_name not in keys:
        raise ValueError(f"Parameter '{parameter_name}' not found in dictionary parameters.")
    
    grids = jnp.meshgrid(*values, indexing='ij')
    param_idx = keys.index(parameter_name)
    param_grid = grids[param_idx]
    param_flat = param_grid.ravel()
    
    return jnp.dot(weights, param_flat)
