import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
from typing import Callable, Tuple, Any

from dmipy_jax.models.flow_fcd import FlowUNet

def get_ot_cfm_loss(model: FlowUNet, x_1: jax.Array, x_0: jax.Array, context: jax.Array, t: jax.Array, key: jax.random.PRNGKey, sigma_min: float = 1e-4):
    """
    Computes/Estimates the OT-CFM loss.
    
    Target path: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
    Target vector field: u_t = x_1 - (1 - sigma_min) * x_0
    
    Args:
        model: Vector field network v(x, t, c).
        x_1: Data sample (Real FLAIR).
        x_0: Noise sample N(0, I).
        context: Conditioning info (T1 stack).
        t: Time points in [0, 1].
        
    Returns:
        loss: MSE || v(x_t, t, c) - u_t ||^2
    """
    # Broadcast t to compatible shape if necessary, usually structured as (Batch,)
    # x shapes: (Batch, C, H, W)
    t_expand = t[:, None, None, None]
    
    # OT Path interpolation
    # t goes from 0 to 1
    # psi_t(x_0) = (1 - (1 - sigma)*t)*x_0 + t*x_1
    # derivative wrt t: -(1-sigma)*x_0 + x_1 = x_1 - (1-sigma)*x_0
    
    alpha_t = 1 - (1 - sigma_min) * t_expand
    beta_t = t_expand
    
    x_t = alpha_t * x_0 + beta_t * x_1
    
    # Target vector field (conditional expectation of flow)
    u_t = x_1 - (1 - sigma_min) * x_0
    
    # Model prediction
    # vmap over batch
    # model call: (x, t, context, key) -> out
    # t is (Batch,), model expects scalar t per sample in vmap?
    # My FlowCNF implementation expects: call(x, t, context) where x is (C,H,W), t is scalar
    
    def apply_model(x_i, t_i, c_i, k_i):
        return model(x_i, t_i, c_i, key=k_i)
        
    keys = jax.random.split(key, x_1.shape[0])
    v_t = jax.vmap(apply_model)(x_t, t, context, keys)
    
    loss = jnp.mean((v_t - u_t)**2)
    return loss

def train_step_ot_cfm(model: FlowUNet, opt_state: optax.OptState, x_1: jax.Array, context: jax.Array, key: jax.random.PRNGKey, optimizer: optax.GradientTransformation):
    """
    Performs a single training step for OT-CFM.
    """
    key_noise, key_time, key_dropout = jax.random.split(key, 3)
    
    batch_size = x_1.shape[0]
    
    # Sample Noise x_0
    x_0 = jax.random.normal(key_noise, x_1.shape)
    
    # Sample Time t uniformly
    t = jax.random.uniform(key_time, (batch_size,))
    
    loss, grads = eqx.filter_value_and_grad(get_ot_cfm_loss)(model, x_1, x_0, context, t, key_dropout)
    
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

# Inference / Sampling
def generate_flair(model: FlowUNet, context: jax.Array, key: jax.random.PRNGKey, sigma_min: float = 1e-4, dt0: float = 0.05):
    """
    Generates FLAIR image from T1 context using Neural ODE.
    Solved from t=0 (Noise) to t=1 (Data).
    
    Args:
        model: Trained Vector Field.
        context: T1 stack (C, H, W).
        
    Returns:
        x_1: Generated FLAIR.
    """
    # Context shape: (C, H, W) -> assumes single sample generation
    # If batch, vmap this function
    
    x_0 = jax.random.normal(key, (1, context.shape[1], context.shape[2])) # Output channel 1
    
    def vector_field(t, x, args):
        # x: State (1, H, W)
        # t: scalar time
        # args: context
        ctx = args
        return model(x, t, ctx)
        
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    # solver = diffrax.Heun() 
    
    # Integrate from 0 to 1
    # OT-CFM matches probability path from Noise (0) to Data (1)
    # So we simply solve ODE dx/dt = v(x,t)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=dt0,
        y0=x_0,
        args=context,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3)
    )
    
    return sol.ys[-1] # Final state at t=1
