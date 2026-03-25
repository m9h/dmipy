
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from functools import partial

from dmipy_jax.reconstruction.csd.deep.model import EquivariantCSD
from dmipy_jax.reconstruction.csd.classic.solvers import _get_response_matrix

def unsupervised_loss(model, x_signal, x_bvecs, response_coeffs, key):
    """
    Auto-Encoder Loss:
    S -> FOD -> S_pred
    """
    # 1. Predict FOD
    # x_signal is SH coeffs of signal (N_vox, N_in) or similar?
    # Assuming input is already SH coeffs.
    fod_pred_sh = model(x_signal) # (45,)
    
    # 2. Convolve with Response to get Signal
    # S_pred_lm = FOD_lm * R_l
    lmax_out = 8 # hardcoded or derive?
    # We need to know lmax of output.
    # Assuming model output is lmax=8 (45 coeffs)
    
    R_diag = _get_response_matrix(response_coeffs, lmax_out)
    
    # Broadcast or Multiply
    # R_diag matches FOD size
    s_pred_sh = fod_pred_sh * R_diag
    
    # 3. Loss
    # MSE in SH domain (equivalent to MSE on sphere by Parseval's, up to factor)
    mse = jnp.mean((s_pred_sh - x_signal)**2) # Padding x_signal if sizes differ?
    # Usually Input Signal has L=4, Output FOD L=8.
    # We can only compare Signal up to L=4 (Input Lmax).
    # So we truncate s_pred_sh to L=4.
    
    n_in = x_signal.shape[0]
    mse = jnp.mean((s_pred_sh[:n_in] - x_signal)**2)
    
    # 4. Regularization
    # Sparsity on FOD? Non-negativity?
    # e3so3 uses specific priors.
    # L1 norm on FOD amps?
    l1 = jnp.mean(jnp.abs(fod_pred_sh))
    
    return mse + 1e-3 * l1

@eqx.filter_jit
def train_step(model, opt_state, x_signal, response_coeffs, optimizer):
    loss_fn = partial(unsupervised_loss, response_coeffs=response_coeffs, x_bvecs=None, key=None)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_signal)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss

def train_loop(data_sh, response_coeffs, key, steps=1000):
    """
    Simple training loop.
    data_sh: (N_vox, N_sh_in)
    """
    # Init Model
    model = EquivariantCSD(key, lmax_in=4, lmax_out=8)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Scan
    def scan_body(carrier, x):
        model, opt_state = carrier
        # Batch? Assuming SGD on single voxel for demo or vmap outside.
        # Actually we need Batch Training.
        pass
    
    # Batch Wrapper
    @eqx.filter_jit
    def train_batch(model, opt_state, batch_sh):
        # Average gradients across batch
        loss_fn_vmap = jax.vmap(partial(unsupervised_loss, response_coeffs=response_coeffs, x_bvecs=None, key=None), in_axes=(None, 0))
        
        def batch_loss(m, x):
            return jnp.mean(loss_fn_vmap(m, x))
            
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model, batch_sh)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Dummy Loop
    print("Agent 2: Starting Unsupervised Training...")
    
    # Data Loader iterator would go here
    # For now, just one batch step
    model, opt_state, loss = train_batch(model, opt_state, data_sh[:32]) # Batch 32
    print(f"Values: Loss={loss}")
    
    return model
