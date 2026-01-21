import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.algebra.initializers import AlgebraicInitializer

def test_algebraic_initializer_kurtosis():
    print("Testing Algebraic Initializer (Kurtosis)...")
    
    # 1. Setup Ground Truth
    D_true = 2.0e-3 # mm^2/s
    K_true = 1.0    # dimensionless
    S0 = 1.0
    
    # b-values in s/mm^2
    b_values = jnp.array([0.0, 1000.0, 2000.0])
    
    # 2. Generate Synthetic Signal (DKI Model)
    # ln(S/S0) = -b*D + (1/6)*b^2*D^2*K
    log_S = -b_values * D_true + (1.0/6.0) * (b_values**2) * (D_true**2) * K_true
    signals = S0 * jnp.exp(log_S)
    
    # 3. Run Initializer
    initializer = AlgebraicInitializer(target_b_indices=(1, 2))
    
    # JIT the inference
    predict_fn = jax.jit(initializer)
    preds = predict_fn(b_values, signals)
    
    D_pred = preds['D']
    K_pred = preds['K']
    
    print(f"Ground Truth: D={D_true}, K={K_true}")
    print(f"Predictions:  D={D_pred}, K={K_pred}")
    
    # 4. Verify
    # We expect exact match (floating point error only) for this exact model
    # since the derivation is exact for the truncation order.
    
    assert jnp.isclose(D_pred, D_true, atol=1e-5), f"D mismatch: {D_pred} vs {D_true}"
    assert jnp.isclose(K_pred, K_true, atol=1e-2), f"K mismatch: {K_pred} vs {K_true}"
    
    print("✅ SUCCESS: Algebraic initializer recovered parameters.")

if __name__ == "__main__":
    test_algebraic_initializer_kurtosis()
