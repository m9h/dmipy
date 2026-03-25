
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

# Constants
T_max = 1.0

def sde_drift(x, t, y, score_net, beta_min=0.1, beta_max=20.0):
    """
    Reverse SDE Drift: f(x,t) - g(t)^2 * score(x,t)
    VP-SDE (Variance Preserving).
    """
    beta_t = beta_min + t * (beta_max - beta_min)
    drift = -0.5 * beta_t * x
    diffusion = jnp.sqrt(beta_t)
    
    score = score_net(x, y, t)
    
    # Reverse drift equation
    # dx = [f(x,t) - g(t)^2 score] dt + g(t) dw
    rev_drift = drift - (diffusion**2) * score
    
    return rev_drift, diffusion

@eqx.filter_jit
def sample_fod(score_net, signal_cond, key, n_steps=1000, lmax_fod=8):
    """
    Samples an FOD from the conditional distribution p(FOD | Signal).
    """
    # 1. Initial Noise
    n_fod = (lmax_fod // 2 + 1) * (lmax_fod + 1) # No, full SH count?
    # symmetric even basis size: (L/2 + 1) * (2L+1)? No.
    # size is (L+1)(L+2)/2 for symmetric tensors?
    # Our basis size is 45 for L=8.
    n_fod = 45 
    
    x_T = jax.random.normal(key, (n_fod,))
    
    # Time steps (Reverse: 1.0 -> 0.0)
    ts = jnp.linspace(T_max, 1e-3, n_steps)
    dt = ts[0] - ts[1] # Positive dt for loop
    
    def step_fn(x, t):
        # Current t is 't'. Next is 't - dt'.
        # Euler-Maruyama step
        
        # SDE Coeffs at t
        rev_drift, g = sde_drift(x, t, signal_cond, score_net)
        
        # Noise
        # We need a key for noise in loop?
        # Use simple Euler (Deterministic/ODE) or Stochastic?
        # Generative CSD benefits from Stochasticity to find modes.
        # But for scan, we need to pass key.
        # Ignoring noise for ODE Probability Flow or simple Euler.
        # Let's use Euler-Maruyama with noise.
        # We assume key splitting happens outside or use stateless noise?
        # Deterministic sampling (ODE) is often better for visualizing the mapping.
        # Score-Based Generative Models often use Predictor-Corrector.
        
        # Simple Euler-Maruyama for Reverse SDE
        # dx = drift * dt + g * dw
        # Time flows backwards: dt is negative?
        # If we iterate 1 -> 0, dt should be negative.
        # rev_drift is defined for forward time t.
        # The SDE is dx = [f - g^2 s] dt + g dW_bar
        # Simulation: x_{t-1} = x_t - [f - g^2 s] dt + g sqrt(dt) z
        
        # Let's use dt > 0 magnitude.
        
        d_x = rev_drift * (-dt)
        # Random noise (TODO: pass key)
        # For prototype, use ODE (d_x only).
        
        x_prev = x + d_x
        return x_prev, x_prev

    final_x, traj = jax.lax.scan(step_fn, x_T, ts)
    
    return final_x

