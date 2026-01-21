
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.fitting.algebraic import dti_algebraic_init

def main():
    print("=== Testing Algebraic Initializers ===")
    
    # 1. Simulate DTI Data
    print("Simulating DTI Signal...")
    b0 = 1000.0
    # True Tensor: Diagonal [1e-3, 0.5e-3, 0.2e-3] rotated?
    # Let's use simple diagonal first to check easy case.
    # D = diag([1.5, 0.5, 0.2]) * 1e-3
    D_true_diag = jnp.array([1.5e-3, 0.5e-3, 0.2e-3])
    # Rotated
    D_true_elements = jnp.array([1.5e-3, 0.0, 0.0, 0.5e-3, 0.0, 0.2e-3]) # Dxx, xy, xz, yy, yz, zz
    
    # Generate acquisition scheme
    n_dirs = 30
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (n_dirs, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = jnp.ones(n_dirs) * 1000.0 # b=1000
    
    # Add b0
    bvecs = jnp.concatenate([jnp.array([[0.,0.,0.]]), bvecs])
    bvals = jnp.concatenate([jnp.array([0.]), bvals])
    
    # Synthesize Signal
    # S = S0 * exp(-b * g^T D g)
    S_sim = []
    for i in range(len(bvals)):
        b = bvals[i]
        g = bvecs[i]
        # D_eff = g^T D g
        # D matrix construction from elements [xx, xy, xz, yy, yz, zz]
        dxx, dxy, dxz, dyy, dyz, dzz = D_true_elements
        D_mat = jnp.array([[dxx, dxy, dxz], [dxy, dyy, dyz], [dxz, dyz, dzz]])
        
        Deff = g.T @ D_mat @ g
        S = b0 * jnp.exp(-b * Deff)
        S_sim.append(S)
    
    S_sim = jnp.stack(S_sim)
    
    # 2. Run Algebraic Inversion
    print("Running Linearized DTI Fit...")
    D_pred = dti_algebraic_init(bvals, bvecs, S_sim)
    
    print(f"True D elements: {D_true_elements}")
    print(f"Pred D elements: {D_pred}")
    
    # Error
    mse = jnp.mean((D_pred - D_true_elements)**2)
    print(f"MSE: {mse:.2e}")
    
    if mse < 1e-12: # Expect near perfect for noiseless
        print("SUCCESS: DTI Algebraic Init accurate.")
    else:
        print("FAILURE: DTI Algebraic Init inaccurate.")
        return

    # 3. Speed Test
    print("Benchmarking Speed...")
    import time
    start = time.time()
    for _ in range(1000):
        _ = dti_algebraic_init(bvals, bvecs, S_sim)
        jax.block_until_ready(_)
    end = time.time()
    avg_time = (end - start) / 1000.0
    print(f"Average Time: {avg_time*1000:.3f} ms")
    
    if avg_time < 0.001:
         print("SUCCESS: Extremely fast (<1ms).")
    else:
         print(f"Speed OK ({avg_time*1000:.3f}ms), JIT compilation might improve it further.")

if __name__ == "__main__":
    main()
