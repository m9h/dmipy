
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Callable, Optional
from jaxtyping import Array, Float
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.zeppelin import Zeppelin

class SSFTTrainer(eqx.Module):
    """
    Self-Supervised Fine-Tuning (SSFT) Trainer.
    
    Fine-tunes a pre-trained network on a specific image volume using 
    physics-informed reconstruction loss and spatial regularization.
    """
    optimizer: optax.GradientTransformation
    
    def __init__(self, learning_rate: float = 1e-4):
        self.optimizer = optax.adam(learning_rate)
        
    def loss_fn(self, 
                network: eqx.Module, 
                data_volume: Float[Array, "X Y Z M"], 
                acquisition: JaxAcquisition,
                lambda_tv: float) -> float:
        """
        Computes the combined Physics + and TV loss.
        """
        # 1. Forward Pass (Batch over volume)
        # Network expects (M,) input. Data is (X,Y,Z,M).
        # We need to flatten special dims or vmap.
        
        # vmap over X, Y, Z
        # net_call(signal) -> params_dict
        vmap_net = jax.vmap(jax.vmap(jax.vmap(network)))
        
        pred_params = vmap_net(data_volume)
        # Returns dict of arrays with shape (X, Y, Z, ...)
        
        # 2. Physics Reconstruction Loss
        # Instantiate model with predicted parameters
        # We need a vmapped version of the Zeppelin model call too?
        # Or just use the model's __call__ if it supports broadcasting.
        # Zeppelin implementation typically expects scalar inputs or broadcastable.
        # Let's check dimensions.
        
        # Reconstruct Signal
        # zeppelin_model call: (bvals, bvecs) -> signal (N_meas)
        # If parameters are (X,Y,Z), output will be (X,Y,Z, N_meas) if broadcast works.
        
        # Reconstruct Signal
        # We define a function that takes single voxel params and returns signal
        # Then we vmap it over X, Y, Z
        
        def single_voxel_model(p_lambda_par, p_lambda_perp, p_mu):
            m = Zeppelin(lambda_par=p_lambda_par, lambda_perp=p_lambda_perp, mu=p_mu)
            return m(acquisition.bvalues, acquisition.gradient_directions)
            
        # Vmap over X, Y, Z
        # pred_params values are (X,Y,Z) or (X,Y,Z,2)
        vmap_model = jax.vmap(jax.vmap(jax.vmap(single_voxel_model)))
        
        signal_recon = vmap_model(
            pred_params['lambda_par'],
            pred_params['lambda_perp'],
            pred_params['mu']
        )
        # signal_recon shape: (X, Y, Z, N_meas)
        
        # MSE Loss
        recon_loss = jnp.mean((data_volume - signal_recon) ** 2)
        
        # 3. Total Variation Regularization
        if lambda_tv > 0:
            tv_loss = 0.0
            # Keys: lambda_par, lambda_perp, fraction, mu
            # We regularize scalar maps. For mu (angles), be careful of periodicity.
            # For now, just regularize diffusivities and fraction.
            
            for key in ['lambda_par', 'lambda_perp', 'fraction']:
                if key in pred_params:
                    # shape (X,Y,Z)
                    pmap = pred_params[key]
                    
                    # Finite difference gradients
                    # jnp.gradient returns a list of gradients per axis
                    grads = jnp.gradient(pmap) # [d/dx, d/dy, d/dz]
                    
                    # L1 norm of gradient magnitude (Isotropic TV) or sum of abs (Anisotropic)
                    # GlobalAMICo used Anisotropic (sum of abs diffs).
                    # Anisotropic TV: sum(|dx|) + sum(|dy|) + sum(|dz|)
                    grad_mag = sum(jnp.sum(jnp.abs(g)) for g in grads)
                    
                    tv_loss += grad_mag
            
            # Normalize TV loss by volume size to keep scale reasonable?
            # Or just rely on lambda weighting.
            # Usually mean TV per voxel.
            vol_size = data_volume.shape[0] * data_volume.shape[1] * data_volume.shape[2]
            tv_loss = tv_loss / vol_size
            
            total_loss = recon_loss + lambda_tv * tv_loss
        else:
            total_loss = recon_loss
            
        return total_loss

    def fit(self, 
            network: eqx.Module, 
            data_volume: Float[Array, "X Y Z M"], 
            acquisition: JaxAcquisition,
            lambda_tv: float = 0.1,
            epochs: int = 100) -> eqx.Module:
        """
        Runs the fine-tuning loop.
        """
        # Filter trainable parameters (arrays of floats)
        # We explicitly want to train weights/biases, not static structures
        params, static = eqx.partition(network, eqx.is_inexact_array)
        
        opt_state = self.optimizer.init(params)
        
        @eqx.filter_jit
        def step(params, static, opt_state, data):
            model = eqx.combine(params, static)
            
            # Compute loss and gradients only wrt params
            loss, grads = jax.value_and_grad(
                lambda p: self.loss_fn(eqx.combine(p, static), data, acquisition, lambda_tv)
            )(params)
            
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = eqx.apply_updates(params, updates)
            return new_params, new_opt_state, loss
            
        # Training Loop
        print(f"Starting SSFT (TV={lambda_tv})...")
        for i in range(epochs):
            params, opt_state, loss = step(params, static, opt_state, data_volume)
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss:.6f}")
                
        # Reconstruct final network
        return eqx.combine(params, static)
