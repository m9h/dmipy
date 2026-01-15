
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dmipy_jax.simulation.monte_carlo import simulate_ground_truth, cylinder_sdf


def test_free_diffusion():
    print("Testing Free Diffusion...")
    
    # 1. Setup Simulation for Free Diffusion (SDF that is never > 0)
    # effectively infinite radius
    def infinite_sdf(pos):
        return -100.0 # Always inside

    def random_init(key, N):
        return jax.random.normal(key, shape=(N, 3)) * 10e-6 # Start near zero
    
    sim_func = simulate_ground_truth(infinite_sdf, random_init)
    
    # 2. Parameters
    D = 3.0e-9 # m^2/s (Diffusivity of free water approx)
    N_particles = 10000
    dt = 1e-4 # 0.1 ms
    
    # Create a PGSE waveform
    delta = 0.010 # 10 ms
    Delta = 0.020 # 20 ms
    G_val = 0.040 # 40 mT/m = 0.04 T/m
    
    # Construct waveform
    # We need a time grid
    T_total = Delta + delta + 0.005 # padding
    N_steps = int(T_total / dt)
    
    # PGSE:
    # 0 to delta: +G
    # Delta to Delta+delta: -G (refocused echo equivalent)
    # Actually, for MC simulation, we simulate the "Spin Echo" condition manually or 
    # use the effective gradient logic.
    # Standard PGSE effective gradient is +G then -G.
    
    waveform = np.zeros((N_steps, 3))
    t = np.arange(N_steps) * dt
    
    # First pulse
    mask1 = (t >= 0.001) & (t < 0.001 + delta)
    waveform[mask1, 0] = G_val
    
    # Second pulse (after Delta time from start of first)
    start2 = 0.001 + Delta
    mask2 = (t >= start2) & (t < start2 + delta)
    waveform[mask2, 0] = -G_val
    
    gradient_waveform = jnp.array(waveform)
    
    # 3. Analytic Solution
    # b-value calculation
    # gamma = 2.6751525e8
    gamma = 2.6751525e8
    b = (gamma * G_val * delta)**2 * (Delta - delta/3)
    analytic_signal = np.exp(-b * D)
    
    # 4. Run Calc
    key = jax.random.PRNGKey(0)
    mc_signal = sim_func(gradient_waveform, D, dt, N_particles, key)
    
    print(f"B-value: {b/1e6:.2f} s/mm^2")
    print(f"Analytic Signal: {analytic_signal:.5f}")
    print(f"MC Signal:       {mc_signal:.5f}")
    
    assert np.abs(mc_signal - analytic_signal) < 0.02, "Free diffusion mismatch!"
    print("Free Diffusion Test Passed!")


def test_cylinder_restriction():
    print("\nTesting Cylinder Restriction (Zero Radius limit)...")
    # If radius -> 0, diffusion perp should be 0.
    # We check if D_perp is restricted.
    
    radius = 2.0e-6 # 2 micron
    
    def cyl_sdf(pos):
        return cylinder_sdf(pos, radius=radius)
        
    def cyl_init(key, N):
        # Initialize uniformly in circle? Or just center is fine for tight restriction check
        # Let's initialize uniform random in a box and discard outside, or just rejection sampling
        # For efficiency, let's just start at center for now, though that changes diffusion time behavior slightly
        # Better: Rejection sample
        pass
        # Simple initialization: spread out, but inside
        # To avoid startup issues, just start at 0. Particles will diffuse to fill.
        return jax.random.normal(key, shape=(N, 3)) * (radius / 5.0)

    sim_func = simulate_ground_truth(cyl_sdf, cyl_init)
    
    # Parameters
    D = 2.0e-9 
    dt = 5e-5 # smaller steps for restriction
    N_particles = 10000
    
    # PGSE Perpendicular to cylinder
    delta = 0.010
    Delta = 0.040 # Long time
    G_val = 0.050 # 50 mT/m
    
    T_total = Delta + delta + 0.005
    N_steps = int(T_total / dt)
    
    waveform = np.zeros((N_steps, 3))
    t = np.arange(N_steps) * dt
    
    mask1 = (t >= 0.001) & (t < 0.001 + delta)
    waveform[mask1, 0] = G_val # Gradients in X (perp to Z-cylinder)
    
    start2 = 0.001 + Delta
    mask2 = (t >= start2) & (t < start2 + delta)
    waveform[mask2, 0] = -G_val
    
    gradient_waveform = jnp.array(waveform)
    
    key = jax.random.PRNGKey(42)
    mc_signal = sim_func(gradient_waveform, D, dt, N_particles, key)
    
    print(f"MC Signal (Restricted): {mc_signal:.5f}")
    
    # Free diffusion reference
    gamma = 2.6751525e8
    b = (gamma * G_val * delta)**2 * (Delta - delta/3)
    free_signal = np.exp(-b * D)
    print(f"Free Signal (Ref):      {free_signal:.5f}")
    
    # Restricted signal should be much higher than free diffusion (less attenuation)
    assert mc_signal > free_signal + 0.1, "Signal did not show restriction!"
    print("Cylinder Restriction Test Passed!")

if __name__ == "__main__":
    test_free_diffusion()
    test_cylinder_restriction()
