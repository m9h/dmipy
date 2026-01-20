import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Optional, Callable, Dict, Any, Union
from tqdm.auto import tqdm
from dmipy_jax.fitting.losses import mse_loss, rician_nll_loss

class StochasticTrainer:
    """
    Trainer for stochastic adjustment of parameters using Optax.
    
    This class manages the training loop for optimizing parameters (or neural networks)
    against a JaxMultiCompartmentModel using stochastic gradient descent (SGD/Adam/etc).
    """
    
    def __init__(self, model, optimizer: optax.GradientTransformation):
        """
        Args:
            model: JaxMultiCompartmentModel instance (or similar callable with same API).
            optimizer: Optax GradientTransformation (e.g. optax.adam(1e-3)).
        """
        self.model = model
        self.optimizer = optimizer
    
    def fit(self, 
            params: Any, 
            acquisition: Any, 
            data: jnp.ndarray, 
            batch_size: int = 32, 
            epochs: int = 100, 
            loss_type: str = 'mse', 
            sigma: Optional[float] = None,
            scales: Optional[jnp.ndarray] = None,
            unwrap_fn: Optional[Callable] = None,
            verbose: bool = True):
        """
        Fits the parameters to the data using the configured optimizer.
        
        Args:
            params: Initial parameters (PyTree).
            acquisition: Acquisition scheme object.
            data: observed data, shape (N_samples, N_measurements) or (N_measurements,).
            batch_size: Size of data batches.
            epochs: Number of training epochs.
            loss_type: 'mse' or 'rician'.
            sigma: Noise standard deviation (required for 'rician' loss).
            scales: Optional array of same shape as params (if flat) to scale parameters.
                    Internal optimization is performed on params / scales.
                    Result is rescaled back.
                    Recommended for mixed-scale parameters (e.g. 1e-9 and 1.0).
            unwrap_fn: Optional callable to extract arguments for model_func from params.
                       Useful if params is a Neural Network transforming inputs to physical params.
            verbose: Whether to show progress bar.
            
        Returns:
            Optimized parameters.
        """
        
        # 1. Normalize Data Shape
        if data.ndim == 1:
            # Single voxel case -> treat as batch of size 1
            data = data[None, :] 
            
        n_samples = data.shape[0]
        
        # 2. Scaling
        if scales is not None:
             # Ensure scales match params structure effectively
             # If params is flat array, scales should be flat array.
             params_internal = params / scales
        else:
             params_internal = params
             scales = 1.0 # Broadcastable
        
        # 3. Define Loss Function
        model_call = self.model.model_func
        
        # Wrapper to unscale if needed
        # If unwrap_fn is present, it acts on internal params? Usually unwrap_fn implies NN.
        # If NN, scales might not apply to weights directly in this simple way, or weights are O(1).
        # We assume 'scales' is for DIRECT parameter fitting.
        
        def effective_unwrap(p):
            p_scaled = p * scales
            if unwrap_fn:
                return unwrap_fn(p_scaled)
            else:
                # Default behavior: assume model_func takes p_scaled directly (or arguments)
                # JaxMultiCompartmentModel.model_func takes (params_flat, acq) where params_flat are args?
                # Actually model_func(params, acq). 
                # If unwrap_fn is None, we pass p_scaled to model_func.
                # However, losses.py handles unwrap_fn.
                # Use a local unwrap to handle scaling.
                return [p_scaled] # Wait, losses.py expects unwrap_fn to return *args.
                # If unwrap_fn is None, losses.py calls model_func(params, acq).
                # So we don't return list here if we want to rely on default.
                
        # To avoid confusion, define loss closure fully here.
        if loss_type == 'mse':
            def loss_fn(p_internal, batch_data):
                p_real = p_internal * scales
                if unwrap_fn:
                    args = unwrap_fn(p_real)
                    pred = model_call(acquisition, *args)
                else:
                    pred = model_call(p_real, acquisition)
                return jnp.mean((batch_data - pred) ** 2)
                
        elif loss_type == 'rician':
            if sigma is None:
                raise ValueError("sigma must be provided for Rician loss.")
            def loss_fn(p_internal, batch_data):
                p_real = p_internal * scales
                if unwrap_fn:
                    args = unwrap_fn(p_real)
                    pred = model_call(acquisition, *args)
                else:
                    pred = model_call(p_real, acquisition)
                
                # Inline Rician logic or call losses? 
                # losses.rician_nll_loss takes model_func.
                # Let's call losses.rician_nll_loss but we provide a "model_func" that accepts p_internal
                # No, easier to just write the loss logic or use the private prediction.
                
                # Re-implementing wrapper for losses.rician_nll_loss is cleaner
                # Define a model wrapper that handles scaling
                def scaled_model_func(p_in, acq):
                    p_r = p_in * scales
                    if unwrap_fn:
                         args = unwrap_fn(p_r)
                         return model_call(acq, *args)
                    else:
                         return model_call(p_r, acq)
                         
                return rician_nll_loss(p_internal, acquisition, batch_data, sigma, scaled_model_func)
        else:
             raise ValueError(f"Unknown loss_type: {loss_type}")

        # 4. Define Update Step
        @jax.jit
        def update_step(params, opt_state, batch_data):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, batch_data)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        # 5. Initialize Optimizer
        opt_state = self.optimizer.init(params_internal)
        
        # 6. Training Loop
        history = []
        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Processing")
            
        params = params_internal
        
        for epoch in iterator:
            epoch_loss = 0.0
            steps = 0
            
            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i+batch_size]
                batch_data = data[idx]
                
                params, opt_state, loss_val = update_step(params, opt_state, batch_data)
                
                # Check for NaNs
                # if jnp.isnan(loss_val):
                #     break 
                
                epoch_loss += loss_val
                steps += 1
                
            avg_loss = epoch_loss / steps
            history.append(avg_loss)
            
            if verbose:
                iterator.set_postfix({"Loss": f"{avg_loss:.4e}"})
                
        self.history = history
        return params * scales
