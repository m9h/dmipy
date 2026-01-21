
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

    def _fit_batch(self, Y_batch: jnp.ndarray, LHS: jnp.ndarray, c_and_lower: tuple, lambda_reg: float, rho: float, constrained: bool):
        """
        Internal batch fitting function (to be JIT-compiled).
        """
        K = self.dict_matrix.shape[1]
        A = self.dict_matrix
        AtY = A.T @ Y_batch # [N_atoms, Batch]
        
        x = jnp.zeros((K, Y_batch.shape[1]))
        z = jnp.zeros_like(x)
        u = jnp.zeros_like(x)
        
        def admm_step(carry, _):
            x, z, u = carry
            rhs = AtY + rho * (z - u)
            x_new = jax.scipy.linalg.cho_solve(c_and_lower, rhs)
            v = x_new + u
            kappa = lambda_reg / rho
            
            if constrained:
                # Enforce non-negativity first
                # Using relu
                v_proj = jnp.maximum(v, 0)
            else:
                v_proj = v
            
            if lambda_reg > 0:
                 # Soft thresholding (L1 prox)
                 z_new = jnp.sign(v_proj) * jnp.maximum(jnp.abs(v_proj) - kappa, 0)
            else:
                 z_new = v_proj
                 
            u_new = u + x_new - z_new
            return (x_new, z_new, u_new), None

        final_carry, _ = jax.lax.scan(admm_step, (x, z, u), None, length=100)
        X_hat, _, _ = final_carry
        return X_hat

    def fit(self, data: jnp.ndarray, lambda_reg: float = 0.0, constrained: bool = True, batch_size: int = 10000):
        """
        Fit the model to the data using ADMM.
        
        Args:
            data: Signal data [N_measurements] or [N_voxels, N_measurements].
            lambda_reg: Regularization parameter (e.g. for L1 sparsity).
            constrained: If True, enforces non-negativity (x >= 0).
            batch_size: Number of voxels to process at once (default 10000).
            
        Returns:
            Estimated weights [..., N_atoms]
        """
        # Handle single voxel case
        if data.ndim == 1:
            data = data[None, :]
            was_1d = True
        else:
            was_1d = False

        # Input data is [N_voxels, N_measurements]
        N_voxels = data.shape[0]
        N_atoms = self.dict_matrix.shape[1]
        
        # Precompute Solver Matrices (Shared across batches)
        needs_admm = constrained or (lambda_reg > 0)
        
        if needs_admm:
            rho = 1.0
            A = self.dict_matrix
            AtA = A.T @ A
            LHS = AtA + rho * jnp.eye(N_atoms)
            c_and_lower = jax.scipy.linalg.cho_factor(LHS)
            
            # Compile the batch runner
            # We use partial/closure to capture static config logic
            # OR pass them as args. Passing as args is cleaner for JIT.
            
            # Helper JIT function
            @jax.jit
            def run_batch(d_batch):
                 # Transpose inside 
                 return self._fit_batch(d_batch.T, LHS, c_and_lower, lambda_reg, rho, constrained)

            # Python Loop for Chunking
            results = []
            num_batches = int(jnp.ceil(N_voxels / batch_size))
            
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, N_voxels)
                
                if i % 10 == 0:
                     print(f"Processing batch {i}/{num_batches}...")

                batch_data = data[start:end]
                batch_res = run_batch(batch_data) # Returns [atoms, batch]
                
                # Transpose back and move to CPU (optional, here we keep on device)
                # If we keep on device, result array grows. 
                # For 1M voxels x 2000 atoms x 4 bytes = 8GB. 
                # Hopefully output fits, but intermediates don't.
                
                results.append(batch_res.T)
                
            X_hat = jnp.concatenate(results, axis=0)

        else:
             # Least Squares (Matrix-based)
             # X = pinv(A) @ Y
             # If pinv is precomputed, we can just matmul.
             # Or use lstsq. Using lstsq batch-wise is safer too.
             
             @jax.jit
             def run_lstsq(d_batch):
                  # d_batch: [B, M] -> Y: [M, B]
                  return jnp.linalg.lstsq(self.dict_matrix, d_batch.T)[0].T

             results = []
             num_batches = int(jnp.ceil(N_voxels / batch_size))
             for i in range(num_batches):
                 start = i * batch_size
                 end = min((i + 1) * batch_size, N_voxels)
                 results.append(run_lstsq(data[start:end]))
                 
             X_hat = jnp.concatenate(results, axis=0)
        
        if was_1d:
            return X_hat[0]
        else:
            return X_hat
