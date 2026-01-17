import jax
import jax.numpy as jnp
from dmipy_jax.signal_models import Zeppelin, Stick

def verify_models():
    print("Verifying Zeppelin and Stick models...")
    
    # Dummy data
    N = 10
    bvals = jnp.ones(N) * 1000.0
    # Random gradients
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (N, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    # Model Parameters
    mu = jnp.array([0.0, 0.0]) # Z-axis
    lambda_par = 1.7e-3
    lambda_perp = 0.2e-3
    
    # 1. Zeppelin
    print("Instantiating Zeppelin...")
    zeppelin = Zeppelin(mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp)
    
    print("Compiling Zeppelin call...")
    # JIT compilation check
    signal_z = jax.jit(zeppelin)(bvals, bvecs)
    print(f"Zeppelin Signal Shape: {signal_z.shape}")
    print(f"Zeppelin Signal Mean: {jnp.mean(signal_z)}")
    
    # 2. Stick
    print("Instantiating Stick...")
    stick = Stick(mu=mu, lambda_par=lambda_par)
    
    print("Compiling Stick call...")
    signal_s = jax.jit(stick)(bvals, bvecs)
    print(f"Stick Signal Shape: {signal_s.shape}")
    print(f"Stick Signal Mean: {jnp.mean(signal_s)}")
    
    print("Verification Complete.")

if __name__ == "__main__":
    verify_models()
