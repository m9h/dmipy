import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.fitting.algebraic_initializers import derive_rational_solution

from dmipy_jax.fitting.algebraic_initializers import derive_rational_solution

def test_algebraic_initializer_monoexponential():
    print("Testing Algebraic Initializer (Mono-Exponential)...")
    
    # 1. Setup Ground Truth
    # Single compartment: f1 * exp(-b * D1)
    f1_true = 0.7  # w1
    D1_true = 2.0e-3 # mm^2/s
    
    # b-values (s/mm^2)
    # Typically need 2 shells for N=1 (4 equations: y0..y3? No)
    # derive_rational_solution for N=1 expects measurements y0, y1...
    # For N=1 (2 unknowns: f, D), we need 2 measurements.
    b_values = (1000.0, 2000.0)
    
    # 2. Generate Synthetic Signal
    # S = f * exp(-b * D)
    signals = jnp.array([
        f1_true * jnp.exp(-b * D1_true) 
        for b in b_values
    ])
    
    # 3. Derive and Run Initializer
    # N=1
    initializer = derive_rational_solution(b_values, n_compartments=1)
    
    # JIT the inference
    predict_fn = jax.jit(initializer)
    preds = predict_fn(signals)
    
    # Output keys should be f_1, D_1
    print("Predicted Keys:", preds.keys())
    
    f1_pred = preds['f_1']
    D1_pred = preds['D_1']
    
    print(f"Ground Truth: f={f1_true}, D={D1_true}")
    print(f"Predictions:  f={f1_pred}, D={D1_pred}")
    
    # 4. Verify
    assert jnp.isclose(f1_pred, f1_true, atol=1e-5), f"f mismatch"
    assert jnp.isclose(D1_pred, D1_true, atol=1e-5), f"D mismatch"
    
    print("SUCCESS: Algebraic initializer recovered Mono-Exp parameters.")

if __name__ == "__main__":
    test_algebraic_initializer_monoexponential()
