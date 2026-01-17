
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.scanner.bloch import simulate_acquisition, bloch_dynamics
from typing import NamedTuple

# Mock Objects
class Phantom(NamedTuple):
    positions: jnp.ndarray
    T1: jnp.ndarray
    T2: jnp.ndarray
    B0_inhom: jnp.ndarray = None

class Sequence(NamedTuple):
    gradients: any # Function or array
    rf: any # Function or array
    dt: float

def test_relaxation():
    print("Testing Relaxation...")
    # Single spin, B=0, checking relaxation
    t1 = 1.0
    t2 = 0.1
    m_init = jnp.array([1.0, 0.0, 0.0]) # Start Transverse
    
    # Args: T1, T2, G=0, B1=0, pos=0, B0=0
    args = (
        t1, t2, 
        lambda t: jnp.array([0.,0.,0.]), 
        lambda t: jnp.array([0.,0.]), 
        jnp.array([0.,0.,0.]), 
        0.0
    )
    
    # dM/dt at t=0
    dM = bloch_dynamics(0.0, m_init, args)
    print(f"Initial dM/dt: {dM}")
    
    # dMx/dt = -Mx/T2 = -1/0.1 = -10
    assert jnp.allclose(dM[0], -10.0), f"Expected dMx/dt=-10, got {dM[0]}"
    
    # Run simulation for T2 decay
    # Re-use simulate_acquisition for convenience?
    # It assumes initial state is [0,0,1] (Equilibrium).
    # To test transverse decay, we'd need an excitation pulse first or modify simulate_acquisition.
    # For unit test, checking derivative is good first step.
    
    # Let's test simulation with a sequence that does nothing (should stay at Equilibrium [0,0,1])
    phantom = Phantom(
        positions=jnp.array([[0.,0.,0.]]),
        T1=jnp.array([1.0]),
        T2=jnp.array([0.1])
    )
    sequence = Sequence(
        gradients=lambda t: jnp.array([0.,0.,0.]),
        rf=lambda t: jnp.array([0.,0.]),
        dt=0.001
    )
    signal = simulate_acquisition(phantom, sequence, duration=1.0)
    print(f"Signal (Equilibrium): {signal}")
    print(f"Signal Abs: {jnp.abs(signal)}")
    # Should be 0 transverse magnetization -> Signal 0
    # Wait, simulate_acquisition returns Mx+iMy. Yes, 0.
    assert jnp.allclose(jnp.abs(signal), 0.0, atol=1e-6)

def test_precession():
    print("\nTesting Precession...")
    # Apply B-field via inhomogeneity or gradient
    # We want to see precession.
    # Start at equilibrium [0,0,1]. Apply 90 deg pulse then acquire?
    # Or just simulate_acquisition implies calculating signal over time?
    # simulate_acquisition returns ONE value at end of duration?
    # No, it returns "Complex sum". Usually signal is acquired over a readout window.
    # The current implementation returns M_final (at t=duration).
    # If we want time-course, we'd need multiple calls or modified return.
    # The Prompt says: "Return the complex sum sum(Mx+iMy)."
    # And "simulate_acquisition(..., duration)".
    # This implies a single point measurement at `duration`.
    
    # So let's test a 90 degree pulse.
    # B1 amplitude for 90 deg in 1ms.
    # gamma * B1 * t = pi/2
    # B1 = pi/2 / (gamma * t)
    
    gamma = 2.6751525e8 # rad/s/T
    duration = 1e-3 # 1ms
    b1_amp = (jnp.pi/2) / (gamma * duration)
    
    phantom = Phantom(
        positions=jnp.array([[0.,0.,0.]]),
        T1=jnp.array([100.0]), # Long T1
        T2=jnp.array([100.0])  # Long T2
    )
    
    # Pulse along x -> M rotates from z to -y (by right hand rule? M x B)
    # dM/dt = M x gamma B
    # B = [B1, 0, 0] (x-axis)
    # M starts [0, 0, 1]
    # Torque = [0, 0, 1] x [B1, 0, 0] = [0, B1, 0] -> +y direction?
    # Wait: z cross x = y.
    # So M tips *up*? No, M is at +z. Torque is +y. It moves towards +y.
    # Rotation axis is +x. +z rotates to +y? (Right hand grip rule on x-axis).
    # Yes. +z -> +y -> -z -> -y.
    # So after 90 deg, M should be [0, 1, 0].
    # Signal Mx+iMy = 0 + 1j = 1j.
    # Magnitude 1.
    
    sequence_90 = Sequence(
        gradients=lambda t: jnp.array([0.,0.,0.]),
        rf=lambda t: jnp.array([b1_amp, 0.0]), # B1 along x
        dt=duration/10.0
    )
    
    sig = simulate_acquisition(phantom, sequence_90, duration=duration)
    print(f"Signal after 90x pulse: {sig}")
    print(f"Magnitude: {jnp.abs(sig)}")
    
    assert jnp.allclose(jnp.abs(sig), 1.0, rtol=1e-2)
    # Check phase (should be +90 deg / +j)
    # M should be roughly [0, 1, 0]
    # Real part ~ 0, Imag part ~ 1
    assert jnp.allclose(sig.real, 0.0, atol=1e-2)
    assert jnp.allclose(sig.imag, 1.0, atol=1e-2)

if __name__ == "__main__":
    # Test 1
    test_relaxation()
    # Test 2
    test_precession()
    print("\nAll tests passed!")
