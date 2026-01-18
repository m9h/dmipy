import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Union
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
             # We need to be careful about model.parameter_cardinality
             # But the operator expects a flat vector/array for the last dim.
             # JaxMultiCompartmentModel takes a DICT typically for `model_func` if using compose?
             # No, `model_func` from `compose_models` takes (params_flat, acquisition).
             # So we are good.
             
             # But wait, JaxMultiCompartmentModel.model_func expects flat array of params.
             # So dummy run:
             # We'll do this lazily or in eval.
             # But we need output_shape for super().__init__
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
                             key: param_name, value: array of values.
                             Structure should match AMICO requirements.
                             
        For now, we implement a simplified version where we iterate over combinations.
        """
        # This needs to align with how legacy AMICO builds atoms.
        # Legacy iterates model names, then parameters.
        # For this task, we will construct a basic atom generator.
        
        atoms = []
        # Logic to build atoms from grid
        # This is complex to generalize perfectly without specific schema, 
        # so we'll implement a 'flat' construction:
        # Assume resolution_grid provides a list of parameter vectors defining the dictionary.
        # OR, we follow the Plan: "Wrap dmipy models... Port AMICO".
        
        # Simplified Dictionary Build:
        # We essentially need a list of parameter vectors p_1, ..., p_K.
        # Then M_i = model_func(p_i).
        pass # To be fleshed out or used with external Matrix construction.
        
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
        
        # Determine dimensions
        # If data is (N_meas,), result is (N_atoms,).
        # If data is (N_vox, N_meas), we want (N_vox, N_atoms).
        # We will wrap the single voxel solver with vmap for efficiency.

        def solve_single_voxel(y_voxel):
            # A: Linear Operator for M
            A = linop.MatrixOperator(M)
            
            # Loss: 1/2 || Ax - y ||^2
            # Prox of this is (A^T A + rho I)^-1 (A^T y + rho v)
            # scico SquaredL2Loss handles this if we use it as 'f' in ADMM
            # provided we check if it supports the linear operator efficiently.
            # Actually, using PGM (FISTA) is often better for L1+Pos problems (Sparse coding).
            # Prox of L1+Pos is analytic. Gradient of L2 is cheap (A^T(Ax-y)).
            
            f = loss.SquaredL2Loss(y=y_voxel, A=A)
            
            # Composite Regularizer: L1 + NonNegative
            # We can define a custom functional or use sum if supported.
            # Simpler: Use PGM with a functional that represents L1 + NonNeg.
            # Prox_{lambda L1 + NonNeg}(v) = ReLU(SoftThresh(v, lambda))
            
            class L1PlusNonNeg(functional.Functional):
                has_eval = True
                has_prox = True
                def __init__(self, alpha):
                    self.alpha = alpha
                    
                def __call__(self, x):
                    return self.alpha * jnp.sum(jnp.abs(x)) # + inf if x<0 technically
                
                def prox(self, v, s):
                    # prox_{s * (alpha L1 + NonNeg)}(v)
                    # effective threshold = s * alpha
                    thresh = s * self.alpha
                    return jnp.maximum(0.0, jnp.sign(v) * jnp.maximum(0.0, jnp.abs(v) - thresh))
            
            g = L1PlusNonNeg(alpha=lambda_l1)
            
            solver = optimize.PGM(
                f=f,
                g=g,
                L0=1.0,
                x0=jnp.zeros(M.shape[1]),
                maxiter=maxiter,
                step_size=None, # Auto-estimate
                history_size=5,
                itstat_view_interval=maxiter+1
            )
            
            return solver.solve()
            
        if y.ndim == 1:
            return solve_single_voxel(y)
        else:
            # y is (N_vox, N_meas). vmap over 0-th axis.
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
        # Note: input_shape is (Nx, Ny, Nz, Nparams) or similar.
        # TV should effectively be sum of TVs for each parameter channel?
        # scico.functional.TotalVariation default sums over specified axes.
        # We assume independent TV for each parameter map for now.
        # We apply TV over spatial axes: typically 0, 1, (2). 
        # The last axis is params.
        
        spatial_axes = tuple(range(len(self.op.input_shape) - 1))
        g = lambda_tv * functional.TotalVariation(axis=spatial_axes)
        
        # 3. Solver
        # Using PGM (FISTA)
        
        # Initial guess: simple starting point (e.g. 0.5)
        # Or better: use a cheap voxel-wise fit if available.
        x0 = jnp.ones(self.op.input_shape) * 0.5
        
        solver = optimize.PGM(
            f=f,
            g=g,
            L0=L0,
            x0=x0,
            maxiter=maxiter,
            itstat_view_interval=maxiter // 5 if maxiter >= 5 else 1,
        )
        
        result = solver.solve()
        return result
