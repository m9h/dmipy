import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Union
import numpy as np
import scico
from scico import functional, linop, loss, operator, optimize
from scico.numpy import BlockArray

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

class MicrostructureOperator(operator.Operator):
    """
    Wraps a JaxMultiCompartmentModel as a SCICO Operator.
    
    This operator maps microstructure parameters (x) to predicted signals (y).
    It supports spatial dimensions (batching).
    """
    def __init__(self, model: JaxMultiCompartmentModel, acquisition, input_shape: Tuple[int, ...]):
        """
        Args:
            model: Instance of JaxMultiCompartmentModel.
            acquisition: Acquisition scheme object.
            input_shape: Shape of the parameter array (..., N_params).
                         Spatial dimensions + parameter dimension.
        """
        self.model = model
        self.acquisition = acquisition
        
        # Determine output structure
        # If input is (N_x, N_y, N_params), output is (N_x, N_y, N_bvals)
        # We need to know N_bvals from acquisition.
        # Assuming acquisition has 'bvals' or similar that defines measurement count.
        # Or we can run a dummy prediction.
        
        # We'll infer N_meas from a dummy call if needed, or check acquisition.
        # Let's assume acquisition.N_measurements exists or similar.
        # If not, we try to deduce.
        if hasattr(acquisition, 'N_measurements'):
             self.N_meas = acquisition.N_measurements
        elif hasattr(acquisition, 'bvals'):
             self.N_meas = len(acquisition.bvals)
        else:
             # Fallback: Run dummy
             dummy_params = jnp.zeros(input_shape[-1]) # Correct size?
             pass

        # For super init, we need input_shape and output_shape.
        # input_shape is passed in.
        # output_shape = input_shape[:-1] + (N_meas,)
        
        # To get N_meas safely:
        # We can't easily run JAX code in __init__ if strictly avoiding compilation,
        # but here it's fine.
        dummy_p = jnp.zeros(input_shape[-1])
        # model.model_func expects 1D params?
        # Yes, standard fit uses flat array.
        try:
             dummy_sig = self.model.model_func(dummy_p, self.acquisition)
             self.N_meas = dummy_sig.shape[0]
        except Exception as e:
             # Sometimes model expects specific ranges/validity.
             # Let's try init_params logic if available or just all ones.
             dummy_p = jnp.ones(input_shape[-1]) * 0.5
             dummy_sig = self.model.model_func(dummy_p, self.acquisition)
             self.N_meas = dummy_sig.shape[0]

        output_shape = input_shape[:-1] + (self.N_meas,)
        
        super().__init__(input_shape=input_shape, output_shape=output_shape, input_dtype=jnp.float32, output_dtype=jnp.float32)

    def _eval(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the operator: A(x) -> y
        """
        # x shape: (..., N_params)
        # model_func expects (N_params,)
        # We can use jax.vmap to handle spatial dims.
        
        # Check dimensionality
        # If x is 1D (single voxel), just call.
        # If x is ND, vmap appropriate number of times or flatten spatial.
        
        ndim_spatial = x.ndim - 1
        
        if ndim_spatial == 0:
            return self.model.model_func(x, self.acquisition)
        else:
            # Construct vmap
            # We want to map over all leading dimensions.
            # Convenient way: flatten, map, reshape.
            x_flat = x.reshape(-1, x.shape[-1])
            
            # Map over batch dimension (0)
            map_func = jax.vmap(lambda p: self.model.model_func(p, self.acquisition))
            y_flat = map_func(x_flat)
            
            # Reshape back: (..., N_meas)
            y_shape = x.shape[:-1] + (self.N_meas,)
            return y_flat.reshape(y_shape)


class AMICOSolver:
    """
    Port of AMICO (Accelerated Microstructure Imaging via Convex Optimization).
    This implementation uses SCICO/JAX for high performance.
    
    Solves: y = M x
    subject to constraints (x >= 0, etc.) and regularization.
    """
    def __init__(self, model: JaxMultiCompartmentModel, acquisition):
        self.model = model
        self.acquisition = acquisition
        self.dictionary = None
        self.scales = None
        
    def generate_dictionary(self, resolution_grid: dict):
        """
        Generates the atom matrix M.
        
        Args:
            resolution_grid: Dictionary defining grid for each parameter.
                             Keys must match the unique parameter names in self.model.parameter_names.
                             Values should be 1D arrays/lists of values to iterate.
                             
        Returns:
            JAX Array of shape (N_meas, N_atoms). Also sets self.dictionary.
        """
        import itertools
        
        full_dictionary_parts = []
        
        # Re-construct parameter name mapping to match JaxMultiCompartmentModel logic
        # We need this to verify which grid entry corresponds to which model parameter.
        
        # Tracking seen names to replicate collision logic
        seen_names = []
        
        for i, sub_model in enumerate(self.model.models):
            # 1. Identify Parameter Names for this sub-model
            # We map: unique_name_in_mcm -> original_name_in_submodel
            sub_model_params = [] # List of (orig_name, unique_name)
            
            for pname in sub_model.parameter_names:
                unique_name = pname
                if unique_name in seen_names:
                     unique_name = f"{pname}_{i+1}"
                seen_names.append(unique_name)
                sub_model_params.append((pname, unique_name))
            
            # 2. Extract Grids for this sub-model
            # We prepare lists of values for itertools.product
            # Order matters: we must pass args/kwargs in correct order to sub_model call.
            
            val_lists = []
            param_order = []
            
            # We need to know which parameters are required by the sub-model.
            # Usually `sub_model.parameter_names` covers inputs.
            # But we must check if they are provided in grid or fixed in instance.
            
            skip_model = False
            
            for orig_name, unique_name in sub_model_params:
                if unique_name in resolution_grid:
                    # Grid provided
                    vals = resolution_grid[unique_name]
                    # Ensure iterable
                    if not isinstance(vals, (list, tuple, jnp.ndarray, np.ndarray)):
                        vals = [vals]
                    val_lists.append(vals)
                    param_order.append(orig_name)
                    
                elif hasattr(sub_model, orig_name) and getattr(sub_model, orig_name) is not None:
                     # Fixed parameter in model definition
                     val = getattr(sub_model, orig_name)
                     val_lists.append([val])
                     param_order.append(orig_name)
                else:
                    # Parameter missing from grid and not fixed?
                    # Check cardinality. If cardinality is for vector (e.g. mu),
                    # we might key it by unique_name in grid.
                    # If missing, we can't generate atoms.
                    # Warn or Error?
                    # For now, assume user provides all necessary grids.
                    print(f"Warning: Parameter {unique_name} not in grid and not fixed. Using defaults if possible.")
                    pass

            if not val_lists:
                # No parameters to vary? Maybe 0-param model?
                # Just call once?
                if not sub_model_params:
                     val_lists = [[]] # One combination: empty
                else:
                     # Missing params
                     continue
            
            # 3. Generate Combinations
            combinations = list(itertools.product(*val_lists))
            
            if not combinations:
                continue
                
            # 4. Batched Signal Generation
            # Convert to JAX arrays for vmap
            # combinations is list of tuples: [ (v1, v2), (v1', v2'), ... ]
            # Transpose to: [ (v1, v1'...), (v2, v2'...) ]
            flat_params_columns = list(zip(*combinations))
            flat_params_arrays = [jnp.array(col) for col in flat_params_columns]
            
            # Wrapper for vmap to unpack args to kwargs
            def eval_sub_model_wrapper(*args):
                kwargs = dict(zip(param_order, args))
                return sub_model(self.acquisition.bvals, self.acquisition.gradient_directions, **kwargs)
            
            if len(flat_params_arrays) > 0:
                # Map over the batch dimension
                batch_signals = jax.vmap(eval_sub_model_wrapper)(*flat_params_arrays)
                # Shape: (N_atoms_for_model, N_meas)
                full_dictionary_parts.append(batch_signals.T) # (N_meas, N_atoms)
            else:
                 # Case 0 params (isotropic fixed?)
                 # Just call once
                 sig = sub_model(self.acquisition.bvals, self.acquisition.gradient_directions)
                 full_dictionary_parts.append(sig[:, None])

        # Concatenate all atoms from all sub-models
        if full_dictionary_parts:
            self.dictionary = jnp.concatenate(full_dictionary_parts, axis=1)
        else:
            self.dictionary = jnp.zeros((getattr(self.acquisition, 'N_measurements', 0), 0))
            
        return self.dictionary
        
    def fit(self, data, dictionary_matrix, lambda_l1=1e-3, lambda_l2=0.0, rho=1.0, maxiter=100):
        """
        Solves the regularized linear inverse problem using SCICO ADMM.
        
        min_x 1/2 ||Mx - y||_2^2 + lambda_1 ||x||_1 + I(x>=0)
        
        Args:
            data: Observed signal (N_vox, N_meas) or (N_meas,)
            dictionary_matrix: Matrix M (N_meas, N_atoms)
            lambda_l1: Weight for L1 regularization (sparsity).
            lambda_l2: Weight for L2 regularization (ridge).
            rho: ADMM penalty parameter.
            maxiter: Maximum iterations.
            
        Returns:
            Computed coefficient vector x.
        """
        M = jnp.array(dictionary_matrix)
        y = jnp.array(data)
        
        # Calculate Lipschitz constant for step size
        # L = ||M||_2^2
        # Use simple power method or full SVD if small
        # For AMICO, M is usually (N_meas, N_atoms). N_meas ~ 100. N_atoms ~ 1000.
        # jnp.linalg.norm(M, 2) computes spectral norm (largest singular value).
        L_spectral = jnp.linalg.norm(M, ord=2)**2
        
        def solve_single_voxel(y_voxel):
            # A: Linear Operator for M
            A = linop.MatrixOperator(M)
            
            # Loss: 1/2 || Ax - y ||^2
            f = loss.SquaredL2Loss(y=y_voxel, A=A)
            
            # Composite Regularizer
            class L1PlusNonNeg(functional.Functional):
                has_eval = True
                has_prox = True
                def __init__(self, alpha):
                    self.alpha = alpha
                    
                def __call__(self, x):
                    return self.alpha * jnp.sum(jnp.abs(x))
                
                def prox(self, v, s):
                    thresh = s * self.alpha
                    return jnp.maximum(0.0, jnp.sign(v) * jnp.maximum(0.0, jnp.abs(v) - thresh))
            
            g = L1PlusNonNeg(alpha=lambda_l1)
            
            solver = optimize.PGM(
                f=f,
                g=g,
                L0=L_spectral, # Use exact Lipschitz constant
                x0=jnp.zeros(M.shape[1]),
                maxiter=maxiter,
                step_size=None,
            )
            
            return solver.solve()
            
        if y.ndim == 1:
            return solve_single_voxel(y)
        else:
            return jax.vmap(solve_single_voxel)(y)

class GlobalOptimizer:
    """
    Performs global reconstruction with spatial regularization (TV).
    """
    def __init__(self, microstructure_operator: MicrostructureOperator):
        self.op = microstructure_operator
        
    def solve_tv(self, data, lambda_tv=0.1, maxiter=100, L0=1e2):
        """
        Solves: min_x 1/2 || A(x) - y ||_2^2 + lambda_tv * TV(x)
        using Accelerated PGM (FISTA).
        
        Args:
            data: Acquired data (..., N_meas).
            lambda_tv: Regularization weight for TV.
            maxiter: Maximum iterations.
            L0: Initial Lipschitz estimate.
            
        Returns:
            Reconstructed parameter map x.
        """
        y = jnp.array(data)
        
        # 1. Data Fidelity f(x) = 1/2 || A(x) - y ||^2
        f = loss.SquaredL2Loss(y=y, A=self.op)
        
        # 2. Regularizer g(x) = lambda_tv * TV(x)
        spatial_axes = tuple(range(len(self.op.input_shape) - 1))
        # Pass input_shape to avoid lazy init jit errors
        g = lambda_tv * functional.AnisotropicTVNorm(axes=spatial_axes, input_shape=self.op.input_shape)
        
        # 3. Solver
        x0 = jnp.ones(self.op.input_shape) * 0.5
        
        solver = optimize.PGM(
            f=f,
            g=g,
            L0=L0,
            x0=x0,
            maxiter=maxiter,
        )
        
        result = solver.solve()
        return result
