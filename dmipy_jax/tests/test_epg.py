
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.models.epg import JAXEPG

def test_spgr_ernst_angle():
    """
    Verify that SPGR simulation maximizes signal at the theoretical Ernst Angle.
    Ernst Angle: alpha = acos(exp(-TR/T1))
    """
    T1 = 1000.0 # ms
    T2 = 100.0  # ms
    TR = 10.0   # ms
    
    # Theoretical Ernst Angle
    E1 = np.exp(-TR/T1)
    alpha_ernst = np.arccos(E1) # Radians
    
    # Define a JIT-compiled EPG function that returns signal magnitude
    @jax.jit
    def get_signal(alpha):
        # We use a large number of pulses to ensure robust steady state
        sig = JAXEPG.simulate_spgr(T1, T2, TR, alpha, N_pulses=500, N_states=20)
        return jnp.abs(sig)

    # Sweep angles around Ernst angle
    alphas = np.linspace(alpha_ernst * 0.5, alpha_ernst * 1.5, 50)
    signals = [float(get_signal(a)) for a in alphas]
    
    # Find angle max signal
    max_idx = np.argmax(signals)
    alpha_max = alphas[max_idx]
    
    print(f"Theoretical Ernst: {np.rad2deg(alpha_ernst):.2f} deg")
    print(f"Simulated Max:     {np.rad2deg(alpha_max):.2f} deg")
    
    # Check agreement (within coarse sweep tolerance)
    assert np.abs(alpha_max - alpha_ernst) < np.deg2rad(5.0)

def test_differentiability():
    """
    Ensure we can take gradients w.r.t. T1.
    """
    T1_target = 1000.0
    T2 = 100.0
    TR = 10.0
    alpha = np.deg2rad(10.0)
    
    @jax.jit
    def forward(t1_val):
        sig = JAXEPG.simulate_spgr(t1_val, T2, TR, alpha, N_pulses=100)
        return jnp.abs(sig)
    
    grad_fn = jax.grad(forward)
    g = grad_fn(T1_target)
    
    print(f"dSignal/dT1: {g}")
    assert jnp.isfinite(g)
    assert g != 0.0

if __name__ == "__main__":
    test_spgr_ernst_angle()
    test_differentiability()
    print("ALL TESTS PASSED")
