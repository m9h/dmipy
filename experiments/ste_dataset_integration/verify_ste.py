import jax
import jax.numpy as jnp
import numpy as np
from ste_acquisition import STEAcquisitionScheme

# Enable x64 as per strict constraints
jax.config.update("jax_enable_x64", True)

def verify_ste_scheme():
    print("Verifying STEAcquisitionScheme...")
    
    # 1. Mock Data
    N = 10
    bvals = jnp.linspace(0, 3000e6, N)
    grads = jnp.ones((N, 3)) / jnp.sqrt(3)
    
    mixing_time_scalar = 1.0 # 1000ms
    delta_scalar = 0.02 # 20ms
    
    # 2. Instantiate
    scheme = STEAcquisitionScheme(
        bvalues=bvals,
        gradient_directions=grads,
        mixing_time=mixing_time_scalar,
        delta=delta_scalar,
        TE=0.100
    )
    
    # 3. Check Shapes and Types
    assert scheme.mixing_time.shape == (N,)
    assert scheme.delta.shape == (N,)
    assert scheme.Delta.shape == (N,)
    
    print("Shapes verified.")
    
    # 4. Verify Physics Constraint: Delta = TM + delta/3
    # Theoretical Delta
    expected_Delta = mixing_time_scalar + delta_scalar / 3.0
    
    # Check first element
    calc_Delta = scheme.Delta[0]
    
    print(f"Input TM: {mixing_time_scalar} s")
    print(f"Input delta: {delta_scalar} s")
    print(f"Expected Delta (TM + delta/3): {expected_Delta:.6f} s")
    print(f"Stored Delta: {calc_Delta:.6f} s")
    
    assert jnp.allclose(scheme.Delta, expected_Delta, atol=1e-9)
    print("Physics constraint verified: Delta = TM + delta/3")
    
    # 5. Check Info Print
    scheme.print_acquisition_info()
    
    print("Verification SUCCESS.")

if __name__ == "__main__":
    verify_ste_scheme()
