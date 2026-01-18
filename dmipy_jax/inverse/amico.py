
import jax
import jax.numpy as jnp
import equinox as eqx
from scico import functional, linop, loss, optimize
from typing import List, Dict, Any, Optional

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
    
    dict_matrix: jnp.ndarray = eqx.field(init=False)
    acquisition: Any = eqx.field(static=True)
    
    def __init__(self, model: Any, acquisition: Any, dictionary_params: Dict[str, jnp.ndarray]):
        """
        Initialize AMICO Solver.
        
        Args:
            model: The microstructure model (must support standard call with params).
            acquisition: Acquisition scheme.
            dictionary_params: Dictionary of parameters for the atoms. 
                               Each value should be an array of values to grid over.
                               The cartesian product of these arrays forms the dictionary atoms.
        """
        self.acquisition = acquisition
        self.dict_matrix = self.generate_kernels(model, acquisition, dictionary_params)
        
    def generate_kernels(self, model, acquisition, dictionary_params) -> jnp.ndarray:
        """
        Generates the dictionary matrix Phi [N_measurements, N_atoms].
        """
        # Create grid of parameters
        keys = list(dictionary_params.keys())
        values = list(dictionary_params.values())
        
        # Create regular grid (cartesian product)
        # Use meshgrid to generate all combinations
        grids = jnp.meshgrid(*values, indexing='ij')
        
        # Flatten grids to list of atoms
        flat_grids = [g.ravel() for g in grids]
        n_atoms = len(flat_grids[0])
        
        # Evaluate model for each atom
        # Define single evaluation function
        def evaluate_atom(atom_params_values):
            # Construct params dict for this atom
            params = {k: v for k, v in zip(keys, atom_params_values)}
            return model(params, acquisition)
            
        # Vectorize over atoms
        # Stack parameters: [N_atoms, N_params]
        stacked_params = jnp.stack(flat_grids, axis=1)
        
        # vmap over the stacked parameters
        # We need to unpack rows of stacked_params back into args for evaluate_atom
        # But evaluate_atom takes a list/tuple of values. 
        
        # Better approach: vmap the model directly if possible, or use a wrapper.
        # Let's use a wrapper that takes the explicit values.
        
        def model_wrapper(p_values):
            p_dict = {k: v for k, v in zip(keys, p_values)}
            return model(p_dict, acquisition)
            
        atoms = jax.vmap(model_wrapper)(stacked_params)
        
        # atoms shape: [N_atoms, N_measurements]
        # We need [N_measurements, N_atoms] for matrix multiplication y = Phi @ x
        return atoms.T

    def fit(self, data: jnp.ndarray, lambda_reg: float = 0.0, constrained: bool = True):
        """
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
        dictionary_op = linop.MatrixOperator(self.dict_matrix)
        
        # Data fidelity term: (1/2) || y - Ax ||_2^2
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
            # x = pinv(Phi) @ y
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
