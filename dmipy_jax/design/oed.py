import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, List, Union, Optional
import copy
from dmipy_jax.acquisition import JaxAcquisition

def get_sensitivity(
    model_func: Callable,
    params: Dict[str, Any],
    acq_params: Dict[str, Any]
) -> jnp.ndarray:
    """
    Computes the Jacobian of the signal with respect to tissue parameters.
    
    Args:
        model_func: The model function to evaluate. Must accept acq_params keys 
                    and params keys as keyword arguments.
        params: Dictionary of tissue parameters {name: value}.
                Values must be differentiable (floats/arrays).
        acq_params: Dictionary of acquisition parameters 
                    {bvalues: ..., gradient_directions: ...}.
                    
    Returns:
        Jacobian matrix of shape (N_measurements, N_params_total).
    """
    param_keys = list(params.keys())
    # Keep as list to handle mixed shapes (scalars vs arrays)
    param_values = [params[k] for k in param_keys]
    
    def wrapped_model(*values):
        # Reconstruct params dict
        current_params = {k: v for k, v in zip(param_keys, values)}
        
        # Prepare acquisition params (handle alias bvalues -> bvals)
        call_kwargs = acq_params.copy()
        if 'bvalues' in call_kwargs:
            call_kwargs['bvals'] = call_kwargs['bvalues']
            # We keep 'bvalues' too just in case, or remove it?
            # Arguments are passed as kwargs. Model takes (bvals, ... **kwargs).
            # If we pass both, 'bvals' is satisfied. 'bvalues' goes to kwargs (ignored).
            
        # Call model with both acquisition and tissue parameters
        return model_func(**call_kwargs, **current_params)
    
    # jacfwd w.r.t all arguments by default if argnums not specified? 
    # No, we must pass argnums=range(len(param_values)) equivalent?
    # Actually, jacfwd(fun) returns a function that returns the Jacobian of fun.
    # If fun takes multiple args, jacfwd returns jacobian w.r.t. the first arg by default.
    # To get all, we use argnums.
    
    J_tuple = jax.jacfwd(wrapped_model, argnums=range(len(param_values)))(*param_values)
    
    # J_tuple contains one array per parameter.
    # Shapes will be (N_meas, *param_shape).
    # We need to flatten param dims and concatenate along the parameter axis (axis 1).
    
    grad_blocks = []
    for j_elem in J_tuple:
        # j_elem shape: (N_meas, ...) or (N_meas,) if param was scalar and output scalar (but output is vector N_meas)
        # If model output is (N_meas,), and param is scalar, j_elem is (N_meas,). 
        # Wait, jacfwd handles scalar param -> (N_meas,).
        # We want (N_meas, 1).
        
        if j_elem.ndim == 1:
            # Case: scalar param, vector output
            j_elem = j_elem[:, None]
        else:
            # Case: array param (Shape S1, S2...), vector output (N_meas)
            # j_elem shape (N_meas, S1, S2...)
            # Flatten all dims except first
            j_elem = j_elem.reshape(j_elem.shape[0], -1)
            
        grad_blocks.append(j_elem)
        
    return jnp.concatenate(grad_blocks, axis=1)

def compute_fisher_information(jacobian: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """
    Computes the Fisher Information Matrix (FIM).
    FIM = (1/sigma^2) * J.T @ J
    
    Args:
        jacobian: Sensitivity matrix (N_meas, N_params).
        sigma: Standard deviation of noise.
        
    Returns:
        FIM matrix (N_params, N_params).
    """
    return (1.0 / sigma**2) * jnp.dot(jacobian.T, jacobian)

def d_optimality_loss(
    trainable_acq_params: Dict[str, jnp.ndarray],
    static_acq_params: Dict[str, Any],
    model_func: Callable,
    target_params: Dict[str, Any],
    sigma: float = 1.0
) -> float:
    """
    Computes the negative log-determinant of the FIM (to minimize).
    
    Args:
        trainable_acq_params: Dict containing 'bvalues' and 'gradient_directions' to optimize.
        static_acq_params: Dict containing fixed params like 'delta', 'Delta'.
        model_func: Differentiable model function.
        target_params: Target tissue parameters.
        sigma: Noise std dev.
        
    Returns:
        Scalar loss value (negative log determinant).
    """
    # Merge acquisition parameters
    acq_params = {**trainable_acq_params, **static_acq_params}
    
    # Normalize gradient directions for calculation
    # (Though optimization step does this, we ensure it here for consistency if needed, 
    # but usually we trust the input. Let's strictly normalize to be safe in the forward pass)
    vecs = acq_params['gradient_directions']
    norms = jnp.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms = jnp.where(norms == 0, 1.0, norms)
    acq_params['gradient_directions'] = vecs / norms
    
    # 1. Compute Sensitivity
    J = get_sensitivity(model_func, target_params, acq_params)
    
    # 2. Compute FIM
    fim = compute_fisher_information(J, sigma)
    
    # 3. Compute Log-Det
    # Add jitter for stability
    fim_len = fim.shape[0]
    fim = fim + jnp.eye(fim_len) * 1e-6
    
    sign, logdet = jnp.linalg.slogdet(fim)
    
    # We want to MAXIMIZE logdet, so we MINIMIZE -logdet
    return -logdet

def optimize_protocol(
    initial_acq: JaxAcquisition,
    model_func: Callable,
    target_params: Dict[str, Any],
    n_steps: int = 100,
    learning_rate: float = 0.1,
    b_min: float = 0.0,
    b_max: float = 10000e6,
    sigma: float = 1.0,
    b_scale: float = 1e9,
    return_history: bool = False
) -> Union[JaxAcquisition, Tuple[JaxAcquisition, Dict[str, List[Any]]]]:
    """
    Optimizes the acquisition protocol (b-values and gradient directions) using Gradient Descent.
    
    Args:
        initial_acq: Starting protocol.
        model_func: Signal model function.
        target_params: Optimize sensitivity for these tissue parameters.
        n_steps: Number of optimization steps.
        learning_rate: Step size.
        b_min: Minimum b-value constraint.
        b_max: Maximum b-value constraint.
        sigma: Assumed noise standard deviation.
        b_scale: Scaling factor for b-values during optimization (internal only).
                 Helps balance gradients between b-values (~1e9) and unit vectors (~1).
                 Default 1e9 (results in b ~ 0-10).
        
    Returns:
        Optimized JaxAcquisition object.
    """
    
    # Separate trainable params
    # Scale b-values down for optimization
    trainable_params_scaled = {
        'bvalues': (initial_acq.bvalues / b_scale).astype(jnp.float32),
        'gradient_directions': initial_acq.gradient_directions.astype(jnp.float32)
    }
    
    static_params = {}
    if initial_acq.delta is not None:
        static_params['delta'] = initial_acq.delta
    if initial_acq.Delta is not None:
        static_params['Delta'] = initial_acq.Delta
        
    # Wrapper for loss that unscales parameters
    def loss_wrapper(trainable_scaled, static, model, target, sig):
        trainable_unscaled = {
            'bvalues': trainable_scaled['bvalues'] * b_scale,
            'gradient_directions': trainable_scaled['gradient_directions']
        }
        return d_optimality_loss(trainable_unscaled, static, model, target, sig)

    # Value-and-grad function
    val_and_grad_fn = jax.value_and_grad(loss_wrapper, argnums=0)
    
    history = {'loss': [], 'bvalues': []}
    
    for i in range(n_steps):
        loss_val, grads = val_and_grad_fn(trainable_params_scaled, static_params, model_func, target_params, sigma)
        
        if return_history:
            history['loss'].append(float(loss_val))
            # Save unscaled b-values copy
            current_bvals = trainable_params_scaled['bvalues'] * b_scale
            history['bvalues'].append(current_bvals)
        
        # Update bvalues (in scaled space)
        trainable_params_scaled['bvalues'] = trainable_params_scaled['bvalues'] - learning_rate * grads['bvalues']
        
        # Update gradients
        trainable_params_scaled['gradient_directions'] = trainable_params_scaled['gradient_directions'] - learning_rate * grads['gradient_directions']
        
        # --- Constraints ---
        
        # 1. Clip b-values (in scaled space)
        trainable_params_scaled['bvalues'] = jnp.clip(trainable_params_scaled['bvalues'], b_min / b_scale, b_max / b_scale)
        
        # 2. Normalize gradient directions
        vecs = trainable_params_scaled['gradient_directions']
        norms = jnp.linalg.norm(vecs, axis=1, keepdims=True)
        norms = jnp.where(norms == 0, 1.0, norms) # Prevent Div/0
        trainable_params_scaled['gradient_directions'] = vecs / norms
        
        if i % 10 == 0:
            pass
            
    # Construct new optimized acquisition
    new_acq = copy.copy(initial_acq)
    # Scale b-values back up
    new_acq.bvalues = trainable_params_scaled['bvalues'] * b_scale
    new_acq.gradient_directions = trainable_params_scaled['gradient_directions']
    
    return (new_acq, history) if return_history else new_acq
