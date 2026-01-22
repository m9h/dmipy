
import jax
import jax.numpy as jnp
# Add project root to path to ensure we can import dmipy_jax
import sys
import os
sys.path.append(os.getcwd())

from dmipy_jax.diffusion.guidance import microstructure_guided_sampling

def dummy_model(x, t):
    # Simple dummy model: epsilon = x * scale + t * scale
    # t is broadcasted to (batch,) so we reshape for broadcasting
    t_reshaped = t.reshape(-1, 1, 1, 1)
    return x * 0.1 + t_reshaped * 0.01

def dummy_validity_fn(x0):
    # Dummy validity: encourage mean to be 0.5
    # J(x) = - ||mean(x) - 0.5||^2
    return -jnp.sum((jnp.mean(x0) - 0.5)**2)

def verify():
    print("Setting up verification...")
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    shape = (batch_size, channels, height, width)
    
    n_steps = 10
    # Linear schedule for alphas
    betas = jnp.linspace(0.0001, 0.02, n_steps)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    
    key = jax.random.PRNGKey(0)
    scale = 1.0
    
    print("Running sampling loop...")
    try:
        sample = microstructure_guided_sampling(
            model=dummy_model,
            validity_fn=dummy_validity_fn,
            shape=shape,
            n_steps=n_steps,
            scale=scale,
            alphas=alphas,
            alpha_bars=alpha_bars,
            key=key
        )
        print("Sampling complete.")
        print(f"Output shape: {sample.shape}")
        
        if sample.shape != shape:
            print("FAILED: Shape mismatch.")
            return

        if jnp.isnan(sample).any():
            print("FAILED: NaNs detected.")
            return
            
        print("VERIFICATION SUCCESSFUL: Output has correct shape and finite values.")
        
    except Exception as e:
        print(f"FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
