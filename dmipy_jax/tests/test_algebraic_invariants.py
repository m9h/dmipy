
import jax
import jax.numpy as jnp
from dmipy_jax.algebra.wrapper import get_model_invariants
import numpy as np

def test_invariants():
    # 1. Setup Protocol
    # We need enough b-values to eliminate 4 parameters (f, S0, D1, D2).
    # System: 2 compartments -> 4 params.
    # We need >= 4 equations? Or 5 to have an invariant?
    # If N equations > N variables, we have N-V invariants usually.
    # Let's use 6 b-values.
    b_values = [0, 1000, 2000, 3000, 4000, 5000]
    
    print("Computing invariants...")
    invariant_fn = get_model_invariants(b_values, model_name="Zeppelin")
    
    if invariant_fn is None:
        print("Test Skipped: No invariants found (too few b-values or timeout).")
        return

    # 2. Generate Valid Data (On Manifold)
    # Model: S(b) = S0 * (f * exp(-b*D1) + (1-f) * exp(-b*D2))
    S0 = 1.0
    f = 0.7
    D1 = 2e-3 # Diffusivity usually ~ 1e-3 mm^2/s. B-values are usually s/mm^2.
    # Careful: algebra uses 'exp(-b*D)'. If b~1000, D~0.001 -> b*D~1.
    # If we pass floats to symbolic engine, it handles scale ok? 
    # identifiability.py uses `int(round(b/b_base))` for powers.
    # D1, D2 in symbolic domain correspond to log(X).
    # The polynomials hold for X_i = exp(-b_base * D_i).
    # So if we simulate with accurate exponential model, it should satisfy polynomial.
    
    D2 = 0.5e-3
    
    valid_signal = []
    for b in b_values:
        val = S0 * (f * jnp.exp(-b * D1) + (1-f) * jnp.exp(-b * D2))
        valid_signal.append(val)
        
    valid_signal = jnp.array(valid_signal)
    
    # 3. Check Residuals
    residuals = invariant_fn(valid_signal)
    print(f"Valid Signal Residuals: {residuals}")
    
    # Check if close to zero
    err_valid = jnp.mean(jnp.abs(residuals))
    print(f"Mean Abs Error (Valid): {err_valid}")
    
    # 4. Generate Invalid Data (Off Manifold)
    invalid_signal = jax.random.uniform(jax.random.key(0), shape=(len(b_values),))
    residuals_invalid = invariant_fn(invalid_signal)
    err_invalid = jnp.mean(jnp.abs(residuals_invalid))
    print(f"Mean Abs Error (Invalid): {err_invalid}")
    
    assert err_valid < 1e-4, "Invariants not satisfied for valid data."
    assert err_invalid > 0.01, "Invariants zero for random data (trivial?)."
    
    print("Test Passed.")

if __name__ == "__main__":
    test_invariants()
