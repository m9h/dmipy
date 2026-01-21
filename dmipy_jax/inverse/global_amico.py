
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from scico import functional, linop, loss, optimize
from typing import List, Dict, Any, Optional, Tuple
from dmipy_jax.inverse.amico import AMICOSolver

class MicrostructureLinearOperator(linop.LinearOperator):
    """
    Linear Operator representing the application of the AMICO dictionary Phi.
    
    A(x) = x @ Phi.T
    
    where:
    x is the coefficient map of shape (H, W, D, N_atoms) or (..., N_atoms)
    Phi is the dictionary matrix of shape (N_measurements, N_atoms)
    Output y is the signal map of shape (H, W, D, N_measurements)
    """
    def __init__(self, dictionary: jnp.ndarray, input_shape: Tuple[int, ...]):
        """
        Args:
            dictionary: Matrix Phi of shape (N_meas, N_atoms).
            input_shape: Shape of the input coefficients (..., N_atoms).
        """
        self.dictionary = dictionary
        
        # Verify shapes
        n_meas, n_atoms = dictionary.shape
        if input_shape[-1] != n_atoms:
            raise ValueError(f"Last dimension of input_shape ({input_shape[-1]}) must match N_atoms ({n_atoms})")
            
        output_shape = input_shape[:-1] + (n_meas,)
        
        super().__init__(
            input_shape=input_shape, 
            output_shape=output_shape,
            input_dtype=dictionary.dtype,
            output_dtype=dictionary.dtype
        )
        
    def _eval(self, x: jnp.ndarray) -> jnp.ndarray:
        # A(x) = x @ Phi^T
        # x: (..., K), Phi: (M, K) -> Phi.T: (K, M)
        # dot: (..., K) . (K, M) -> (..., M)
        return jnp.dot(x, self.dictionary.T)
        
    def _adj(self, y: jnp.ndarray) -> jnp.ndarray:
        # A^*(y) = y @ Phi
        # y: (..., M), Phi: (M, K)
        # dot: (..., M) . (M, K) -> (..., K)
        return jnp.dot(y, self.dictionary)


class GlobalAMICOSolver(AMICOSolver):
    """
    AMICO Solver with Global Total Variation (TV) Regularization.
    
    Solves:
    min_x 1/2 || Phi @ x - y ||_2^2 + lambda_tv ||D x||_1 + lambda_l1 ||x||_1 + I(x >= 0)
    
    Using ADMM with splitting.
    """
    
    def generate_kernels(self, model, acquisition, dictionary_params) -> jnp.ndarray:
        """
        Generates the dictionary matrix (kernels) for global reconstruction.
        Overrides the base implementation to robustly handle vector parameters and parameter aliasing.
        """
        import itertools
        
        full_dictionary_parts = []
        seen_names = []
        
        # If the model is a JaxMultiCompartmentModel, iterate over sub-models
        # Otherwise treat as single model (fallback)
        if hasattr(model, 'models'):
            sub_models = model.models
        else:
            sub_models = [model]
            
        for i, sub_model in enumerate(sub_models):
            # 1. Identify Parameter Names
            sub_model_params = []
            
            # Use parameter_names from the sub_model
            pnames = getattr(sub_model, 'parameter_names', [])
            
            for pname in pnames:
                unique_name = pname
                # Mimic MCM behavior: if name seen, append suffix _1, _2...
                # Note: This logic must match how MCM constructs names if we want to match user keys.
                # However, AMICOSolver usually receives dictionary_params keyed by unique names.
                if unique_name in seen_names:
                     unique_name = f"{pname}_{i+1}"
                     # If that collides (unlikely if unique_names is consistent), increment
                     # But simple logic: MCM usually handles this mapping.
                
                # We assume dictionary_params uses these unique names.
                seen_names.append(unique_name)
                sub_model_params.append((pname, unique_name))
            
            # 2. Extract Grids
            val_lists = []
            param_order = []
            
            for orig_name, unique_name in sub_model_params:
                if unique_name in dictionary_params:
                    vals = dictionary_params[unique_name]
                    # Ensure iterable list
                    if isinstance(vals, (jnp.ndarray, np.ndarray)) and vals.ndim > 1:
                        # Array of vectors (e.g. mu)
                        vals = [v for v in vals]
                    elif isinstance(vals, (jnp.ndarray, np.ndarray)):
                        vals = vals.tolist()
                    elif not isinstance(vals, (list, tuple)):
                        vals = [vals]
                    
                    val_lists.append(vals)
                    param_order.append(orig_name)
                elif hasattr(sub_model, orig_name) and getattr(sub_model, orig_name) is not None:
                    # Fixed parameter
                    val = getattr(sub_model, orig_name)
                    val_lists.append([val])
                    param_order.append(orig_name)
                else:
                    # Missing parameter. 
                    # If it's optional or handled by kwargs defaults?
                    # warning?
                    pass

            if not val_lists:
                 # No parameters? Call once with defaults
                 val_lists = [[]]
            
            # 3. Generate Combinations
            combinations = list(itertools.product(*val_lists))
            
            if not combinations:
                continue
            
            # 4. Batched Signal Generation
            # Transpose combinations to columns
            flat_params_columns = list(zip(*combinations))
            flat_params_arrays = [jnp.array(col) for col in flat_params_columns]
            
            def eval_wrapper(*args):
                kwargs = dict(zip(param_order, args))
                
                # Inject acquisition timing if available
                if getattr(acquisition, 'delta', None) is not None:
                    kwargs.setdefault('small_delta', acquisition.delta)
                if getattr(acquisition, 'Delta', None) is not None:
                    kwargs.setdefault('big_delta', acquisition.Delta)
                
                # Use acquisition bvals/bvecs/etc.
                return sub_model(acquisition.bvalues, acquisition.gradient_directions, **kwargs)
                
            if len(flat_params_arrays) > 0:
                atoms = jax.vmap(eval_wrapper)(*flat_params_arrays)
                # atoms: (N_atoms, N_meas)
                full_dictionary_parts.append(atoms.T)
            else:
                # No params to vary
                kwargs = {}
                if getattr(acquisition, 'delta', None) is not None:
                    kwargs.setdefault('small_delta', acquisition.delta)
                if getattr(acquisition, 'Delta', None) is not None:
                    kwargs.setdefault('big_delta', acquisition.Delta)
                    
                sig = sub_model(acquisition.bvalues, acquisition.gradient_directions, **kwargs)
                full_dictionary_parts.append(sig[:, None])
                
        # Concatenate
        if full_dictionary_parts:
            # Check shape consistency
            # If some part has different N_meas (unlikely if shared acq), warn?
            dictionary = jnp.concatenate(full_dictionary_parts, axis=1)
        else:
            dictionary = jnp.zeros((len(acquisition.bvalues), 0))
            
        return dictionary

    def fit_global(self, 
                   data: jnp.ndarray, 
                   lambda_tv: float = 0.1, 
                   lambda_l1: float = 0.0, 
                   rho: float = 1.0, 
                   maxiter: int = 50,
                   display: bool = False) -> jnp.ndarray:
        """
        Fit the model globally using ADMM with TV regularization.
        
        Args:
            data: Observed DWI data (H, W, D, N_meas) or (..., N_meas).
                  Spatial dimensions are inferred from data.shape[:-1].
            lambda_tv: Regularization strength for Total Variation.
            lambda_l1: Regularization strength for L1 sparsity.
            rho: ADMM penalty parameter.
            maxiter: Maximum number of iterations.
            display: Whether to print solver convergence stats.
            
        Returns:
            Estimated coefficients x of shape (..., N_atoms).
        """
        # Ensure data is an array
        y = jnp.array(data)
        
        # Get shapes
        n_meas = self.dict_matrix.shape[0]
        n_atoms = self.dict_matrix.shape[1]
        spatial_shape = y.shape[:-1]
        
        if y.shape[-1] != n_meas:
            raise ValueError(f"Data last dimension {y.shape[-1]} does not match dictionary measurements {n_meas}")
            
        x_shape = spatial_shape + (n_atoms,)
        
        # 1. Define Operators
        
        # A: Linearity (Dictionary)
        A = MicrostructureLinearOperator(self.dict_matrix, input_shape=x_shape)
        
        # f: Data Fidelity = 1/2 || Ax - y ||^2
        f = loss.SquaredL2Loss(y=y, A=A)
        
        # 2. Define Regularizers (g functions) and their linear operators (C matrices)
        g_list = []
        C_list = []
        
        # g1: Total Variation
        # TV corresponds to L1 norm of the gradient.
        # functional.AnisotropicTVNorm creates a functional h(x) = sum |D_i x|.
        # But ADMM expects g(z) s.t. z = C x.
        # So we use L1Norm for g, and FiniteDifference for C.
        
        if lambda_tv > 0:
            # Check spatial dimensions
            if len(spatial_shape) < 1:
                # No spatial dimensions -> standard elastic net
                pass
            else:
                # Identify spatial axes. The last axis is atoms (channels), we don't TV over that.
                # Spatial axes are 0 to ndim-2.
                # Construct FiniteDifference operator for spatial axes.
                # circ=False implies Neumann boundary conditions usually (zero grad at boundary?)
                # We apply TV channel-by-channel? Or coupled?
                # Usually TV determines edges. 
                # "Vector-valued TV" / Color TV: sum of norms? Or norm of vectors?
                # Anisotropic TV: Sum of L1 norms of gradients of each channel.
                # This corresponds to standard L1Norm on the output of FiniteDifference.
                
                spatial_dims = tuple(range(len(spatial_shape)))
                
                # Finite Difference Operator:
                # Outputs a BlockArray of gradients.
                # Input: (..., N_atoms)
                # Output: BlockArray of (..., N_atoms) for each spatial dim.
                # We need to reshape/be careful.
                # scico.linop.FiniteDifference supports input_shape.
                C_tv = linop.FiniteDifference(
                    input_shape=x_shape,
                    axes=spatial_dims,
                    circular=False 
                )
                
                g_tv = lambda_tv * functional.L1Norm()
                
                g_list.append(g_tv)
                C_list.append(C_tv)
                
        # g2: Sparsity (L1) + Non-Negativity
        # We can combine them into a single functional that acts on Identity transform.
        # Prox of (lambda * |x| + I(x>=0)) is: ReLU(soft_thresh(x, lambda))
        
        class L1PlusNonNeg(functional.Functional):
            has_eval = True
            has_prox = True
            
            def __init__(self, alpha):
                self.alpha = alpha
                
            def __call__(self, x):
                # val = alpha * sum(|x|) + inf if x<0
                is_neg = jnp.any(x < 0)
                # In functional evaluation we return inf if constraints violated
                # But jax.lax.cond is better. 
                # For ADMM trace, exact value matters.
                # Just simplified:
                reg_val = self.alpha * jnp.sum(jnp.abs(x))
                return jnp.where(is_neg, jnp.inf, reg_val)
                
            def prox(self, v, s, v0=None):
                # Prox of alpha * L1 + I(>=0) with step s
                # argmin 1/2||x-v||^2 + s*(alpha*|x| + I(x>=0))
                # Threshold is s * alpha
                shift = s * self.alpha
                # Soft thresholding
                # x = sign(v) * max(|v| - shift, 0)
                # Then project to >= 0
                soft = jnp.sign(v) * jnp.maximum(jnp.abs(v) - shift, 0.0)
                return jnp.maximum(0.0, soft)
                
        g_sia = L1PlusNonNeg(alpha=lambda_l1)
        C_sia = linop.Identity(x_shape)
        
        g_list.append(g_sia)
        C_list.append(C_sia)
        
        # 3. Solver Setup
        rho_list = [rho] * len(g_list)
        
        # Initial guess
        x0 = jnp.zeros(x_shape)
        
        # We use default CG for x-update since inverting (A^T A + rho sum C^T C) is hard 
        # due to C_tv coupling space and A coupling channels.
        # Scico handles this automatically if x_step not provided.
        
        solver = optimize.ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            x0=x0,
            maxiter=maxiter,
            itstat_options={'display': display}
        )
        
        # Solve
        x_hat = solver.solve()
        
        return x_hat
