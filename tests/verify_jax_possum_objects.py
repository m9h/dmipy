import sys
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

# Add src to path to import dmipy_jax
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dmipy_jax.simulation.scanner.objects import IsochromatPhantom, TrapezoidalGradient

def verify_phantom():
    print("Verifying IsochromatPhantom...")
    N_spins = 1000
    
    # Create dummy data
    positions = jnp.zeros((N_spins, 3))
    T1 = jnp.ones(N_spins) * 1000.0
    T2 = jnp.ones(N_spins) * 100.0
    M0 = jnp.ones(N_spins)
    off_resonance = jnp.zeros(N_spins)
    
    phantom = IsochromatPhantom(
        positions=positions,
        T1=T1, 
        T2=T2, 
        M0=M0, 
        off_resonance=off_resonance
    )
    
    # Check strict typing by attempting to pass wrong shape (this depends on if we enabled runtime checking or just rely on static analysis, 
    # but Equinox modules don't auto-validate shapes unless we added __check_init__ which I did)
    try:
        IsochromatPhantom(
            positions=jnp.zeros((N_spins, 2)), # Wrong shape
            T1=T1, T2=T2, M0=M0, off_resonance=off_resonance
        )
        print("FAIL: Should have raised ValueError for wrong position shape")
    except ValueError as e:
        print(f"PASS: Caught expected error: {e}")
        
    print("IsochromatPhantom verification complete.")

def verify_gradient():
    print("\nVerifying TrapezoidalGradient...")
    
    grad = TrapezoidalGradient(
        amplitude=jnp.array([10.0, 0.0, 0.0]),
        flat_time=jnp.array(10.0),
        ramp_time=jnp.array(5.0),
        start_time=jnp.array(0.0)
    )
    
    # Test at characteristic points
    # t=0 (start) -> 0
    # t=5 (end of ramp up) -> 10
    # t=10 (middle of flat) -> 10
    # t=15 (start of ramp down) -> 10
    # t=20 (end of ramp down) -> 0
    
    times = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    grads = jax.vmap(grad.get_gradients)(times)
    
    expected_x = np.array([0.0, 10.0, 10.0, 10.0, 0.0, 0.0])
    
    print("Gradients at checkpoints:", grads[:, 0])
    
    if jnp.allclose(grads[:, 0], expected_x, atol=1e-5):
        print("PASS: Gradient profile matches expected trapezoid.")
    else:
        print("FAIL: Gradient profile mismatch.")
        
    print("TrapezoidalGradient verification complete.")
    
    # Test JIT
    print("\nTesting JIT compilation...")
    @jax.jit
    def compute_grad(t):
        return grad.get_gradients(t)
        
    val = compute_grad(jnp.array(7.5))
    print(f"JIT run result: {val}")
    print("PASS: JIT compilation successful.")

if __name__ == "__main__":
    verify_phantom()
    verify_gradient()
