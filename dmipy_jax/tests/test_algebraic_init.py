
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.fitting.algebraic_initializers import derive_rational_solution

def test_algebraic_initialization():
    print("=== Testing Algebraic Initialization (JAX) ===")
    
    # 1. Define Protocol
    # Mono-exponential: 2 shells (b=0, b=1000)
    b_values = [0.0, 1000.0]
    
    # 2. Derive Initializer (Symbolic Phase)
    print("Deriving solution...")
    initializer = derive_rational_solution(b_values, n_compartments=1)
    
    print("Parameters found:", initializer.param_names)
    assert 'f_1' in initializer.param_names
    assert 'D_1' in initializer.param_names
    
    # 3. Create Synthetic Data
    # True Params
    f_true = 1000.0 # S0 effectively
    D_true = 2.0e-3 # mm^2/s (if b in s/mm^2)
    
    # Model: S = f * exp(-b * D)
    signals = jnp.array([
        f_true * jnp.exp(-b * D_true) for b in b_values
    ])
    print(f"Synthetic Signals: {signals} (True D={D_true}, f={f_true})")
    
    # 4. Run Initializer (JAX Phase)
    print("Compiling and running...")
    
    # JIT the initializer call
    @jax.jit
    def predict(y):
        return initializer(y)
        
    estimates = predict(signals)
    print("Estimates:", estimates)
    
    f_est = estimates['f_1']
    D_est = estimates['D_1']
    
    # 5. Check Accuracy
    print(f"Estimated f: {f_est:.4f}")
    print(f"Estimated D: {D_est:.6f}")
    
    tol = 1e-5
    if abs(f_est - f_true) < tol and abs(D_est - D_true) < tol:
        print("SUCCESS: Exact recovery.")
    else:
        print("FAILURE: Estimates diverge.")
        
    # 6. Check Batched Execution (vmap)
    print("\nTesting vmap...")
    signals_batch = jnp.stack([signals, signals * 0.5])
    
    @jax.jit
    def predict_batch(y_batch):
        return jax.vmap(initializer)(y_batch)
        
    batch_res = predict_batch(signals_batch)
    print("Batch Estimates f_1:", batch_res['f_1'])
    
    assert batch_res['f_1'].shape == (2,)
    print("SUCCESS: vmap compatibility confirmed.")

if __name__ == "__main__":
    test_algebraic_initialization()
