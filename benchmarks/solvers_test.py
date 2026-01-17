import jax
import jax.numpy as jnp
from dmipy_jax.core.solvers import BlochSimulator, solve_diffusion_sde
import matplotlib.pyplot as plt
import numpy as np

def test_bloch_simulation():
    print("Testing Bloch Simulation...")
    # Parameters
    T1 = 1.0 # 1s
    T2 = 0.1 # 100ms
    m0 = jnp.array([0.0, 0.0, 1.0])
    
    simulator = BlochSimulator(T1, T2, m0)
    
    # 1. Free Induction Decay (no gradient)
    # Start on X axis
    m_init = jnp.array([1.0, 0.0, 0.0])
    
    def zero_gradient(t):
        return jnp.array([0.0, 0.0, 0.0])
        
    pos = jnp.array([0.0, 0.0, 0.0])
    
    t_end = 0.05 # 50ms
    sol = simulator(
        (0.0, t_end),
        m_init,
        zero_gradient,
        pos,
        dt0=1e-4
    )
    
    m_final = sol.ys[-1]
    
    # Check Analytical Solution for T2 decay
    m_analytical_mag = jnp.exp(-t_end / T2)
    m_sim_mag = jnp.linalg.norm(m_final[:2]) # Transverse magnitude
    
    print(f"Time: {t_end*1000} ms")
    print(f"Analytical Transverse M: {m_analytical_mag:.6f}")
    print(f"Simulated Transverse M: {m_sim_mag:.6f}")
    
    assert jnp.allclose(m_sim_mag, m_analytical_mag, rtol=1e-3)
    print("Bloch test passed!")

def test_diffusion_sde():
    print("\nTesting Diffusion SDE...")
    # Parameters
    D = 3.0e-9 # m^2/s (approx water at 37C is 3.0e-9)
    # Actually water is ~3e-9 m^2/s? No, 3e-9 m^2/s = 3000 um^2/s.
    # Water at room temp is ~2.3e-9 m^2/s = 2.3 um^2/ms * 1e-3?
    # 1 um^2/ms = 1e-12 m^2 / 1e-3 s = 1e-9 m^2/s.
    # So D=3e-9 m^2/s = 3 um^2/ms.
    
    # Let's work in SI units for code consistency.
    D_si = 1.0 # m^2/s (Simple number for checking math)
    
    def zero_drift(t, y, args):
        return jnp.zeros_like(y)
        
    def constant_diffusion(t, y, args):
        # returns sigma such that sigma*sigma^T = 2D
        # For scalar diffusion D, sigma = sqrt(2D)
        # We need to return matrix or scale for dW.
        # diffrax ControlTerm with VirtualBrownianTree:
        # if diffusion returns (N, 3), and dW is (N, 3), it does elementwise mult?
        # NO.
        # If we use ControlTerm(diffusion_func, brownian), the result is diffusion_func(t, y, args) * dW(t)
        # We want independent diffusion in each dim.
        # If output is (N, 3) and dW is (N, 3), we probably want elementwise.
        # But ControlTerm usually mimics matrix multiplication.
        # Let's check solve_diffusion_sde implementation.
        # It takes diffusion output and dW.
        # We need to ensure we return something compatible with Elementwise multiplication if we want independent noise.
        # BUT diffrax defaults to matrix multiplication if shapes align?
        # Wait, if y is (N, 3), drift is (N, 3).
        # dW from VirtualBrownianTree(shape=(N, 3)) returns (N, 3).
        # ControlTerm: prod(diffusion(t,y), control(t)).
        # If diffusion returns scalar, it works.
        # If diffusion returns (N, 3), it might do dot product?
        # Let's just return scalar for isotropic diffusion.
        return jnp.sqrt(2 * D_si)
    
    N_particles = 1000
    y0 = jnp.zeros((N_particles, 3))
    
    t_end = 1.0 # 1s
    
    # We need to wrap simulating multiple particles. 
    # solve_diffusion_sde is written for a single state vector y0.
    # If y0 is (N, 3), drift returns (N, 3).
    # dW is (N, 3).
    # constant_diffusion returns scalar.
    # Product: scalar * (N, 3) -> (N, 3). This is valid.
    
    sol = solve_diffusion_sde(
        (0.0, t_end),
        y0,
        zero_drift,
        constant_diffusion,
        dt0=1e-2,
        key=jax.random.PRNGKey(42)
    )
    
    y_final = sol.ys[-1] # (N, 3)
    
    # MSD
    msd = jnp.mean(jnp.sum(y_final**2, axis=1))
    
    # Expected MSD = 6 * D * t
    expected_msd = 6 * D_si * t_end
    
    print(f"Simulated MSD: {msd:.4f}")
    print(f"Expected MSD: {expected_msd:.4f}")
    
    # Stochastic variance is high, check if within 10%
    error = jnp.abs(msd - expected_msd) / expected_msd
    print(f"Error: {error:.2%}")
    
    assert error < 0.1
    print("SDE test passed!")

if __name__ == "__main__":
    test_bloch_simulation()
    test_diffusion_sde()
