import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
import diffrax
from typing import Callable

# --- BENCH Import Setup ---
# Add local bench directory to path if present
if os.path.exists("bench"):
    sys.path.append(os.path.abspath("bench"))

try:
    import bench.diffusion_models as bench_models
    print("BENCH imported successfully.")
except ImportError as e:
    print(f"Error importing BENCH: {e}")
    # Fallback: maybe it's inside bench/bench?
    if os.path.exists("bench/bench"):
        sys.path.append(os.path.abspath("bench"))
        try:
            import bench.diffusion_models as bench_models
            print("BENCH imported from bench/bench.")
        except ImportError as e2:
            print(f"Critical Error: Could not import BENCH. {e2}")
            sys.exit(1)
    else:
        sys.exit(1)

from dmipy_jax.simulation.scanner.bloch import simulate_signal, BlochTorreyGeometry

# --- Configuration ---
B_VALUES = [0, 1000, 2000, 3000] # s/mm^2
DIRECTIONS_PER_SHELL = 30
F_INTRA = 0.6
D_INTRA = 1.7e-9 # m^2/s
D_ISO = 3.0e-9   # m^2/s
T1 = 1.0 # s
T2 = 100.0 # s (Long T2 to match BENCH)
GAMMA_HZ = 42.576e6 # Hz/T
GAMMA_RAD = 2 * np.pi * GAMMA_HZ # rad/s/T

# --- Waveform Definition (Pure JAX) ---
def pgse_waveform(t, G_vec, delta, Delta):
    """
    Defines a PGSE gradient waveform G(t).
    
    Args:
        t: Time (scalar)
        G_vec: Gradient vector (3,) [T/m]
        delta: Pulse duration [s]
        Delta: Pulse separation [s]
        
    Returns:
        Gradient vector at time t (3,)
    """
    # Block pulse 1: [0, delta]
    # Block pulse 2: [Delta, Delta + delta] (Effective -G due to refocusing)
    # Actually, Bloch-Torrey simulates physical magnetization.
    # Refocusing pulse flips phase.
    # Standard approach: Simulate effective gradient G then -G.
    
    # Pulse 1: 0 <= t < delta -> G
    # Gap: delta <= t < Delta -> 0
    # Pulse 2: Delta <= t < Delta + delta -> -G
    
    in_pulse1 = (t >= 0) & (t < delta)
    in_pulse2 = (t >= Delta) & (t < Delta + delta)
    
    # Using jax.lax.select/cond or simple arithmetic
    # G_out = G_vec * in_pulse1 - G_vec * in_pulse2
    
    scale = jnp.where(in_pulse1, 1.0, jnp.where(in_pulse2, -1.0, 0.0))
    return G_vec * scale

# --- Parameters Calculation ---
def calc_G_amp(b_val, delta, Delta):
    """Calculates G amplitude for PGSE (Block pulses)."""
    if b_val == 0:
        return 0.0
    
    b_si = b_val * 1e6 # s/mm^2 -> s/m^2
    gamma = 267.51525e6 # rad/s/T
    
    # b = gamma^2 G^2 delta^2 (Delta - delta/3)
    term = b_si / (Delta - delta/3.0)
    G = jnp.sqrt(term) / (gamma * delta)
    return G

# --- Vmapped Simulation ---
# We define a function that takes geometry and PGSE params, and returns signal
@eqx.filter_jit
def simulate_pgse_shell(geometry, b_val, directions, delta, Delta):
    """
    Simulates a whole shell of directions using vmap.
    """
    G_amp = calc_G_amp(b_val, delta, Delta)
    duration = Delta + delta + 0.001 # Small buffer
    
    # Wrapper for single direction
    def sim_one_dir(direction):
        G_vec = direction * G_amp
        
        # Closure for waveform
        def waveform(t):
            return pgse_waveform(t, G_vec, delta, Delta)
            
        M_final = simulate_signal(geometry, waveform, duration, M0=jnp.array([1., 0., 0.]))
        return jnp.linalg.norm(M_final[:2]) # Transverse magnitude
        
    # Vmap over directions
    # geometry is broadcasted? No, geometry is constant for the shell.
    # directions is (N, 3) mapped.
    signals = jax.vmap(sim_one_dir)(directions)
    return signals

def generate_directions(n_dirs: int) -> np.ndarray:
    indices = np.arange(0, n_dirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_dirs)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack([x, y, z], axis=1)

def run_bench_validation():
    print("--- Starting Optimization Validation ---")
    directions_np = generate_directions(DIRECTIONS_PER_SHELL)
    directions_jax = jnp.array(directions_np)
    
    delta = 20e-3
    Delta = 40e-3
    
    results_bench = []
    results_diffrax = []
    
    t_start_total = time.time()
    
    for b_val in B_VALUES:
        print(f"Processing Shell b={b_val}...")
        
        # --- BENCH ---
        t0_bench = time.time()
        
        # BENCH setup
        if b_val == 0:
            dirs_bench = directions_np[0:1] # Just one point needed
            b_bench_val = 0.0
        else:
            dirs_bench = directions_np
            # BENCH units: b in ms/um^2. 1 s/mm^2 = 0.001 ms/um^2.
            b_bench_val = b_val / 1000.0 
            
        b_bench_arr = np.full(len(dirs_bench), b_bench_val)
        
        # Stick (along Z=0,0)
        s_stick_bench = bench_models.stick(b_bench_arr, dirs_bench, d_a=1.7, theta=0.0, phi=0.0)
        s_ball_bench = bench_models.ball(b_bench_arr, dirs_bench, d_iso=3.0)
        s_total_bench = F_INTRA * s_stick_bench + (1 - F_INTRA) * s_ball_bench
        
        # Broadcast b=0 result if needed to match diffrax shape (if diffrax runs all 30)
        if b_val == 0:
             s_total_bench = np.full(DIRECTIONS_PER_SHELL, s_total_bench[0])
             
        t1_bench = time.time()
        print(f"  BENCH Time: {t1_bench - t0_bench:.5f} s")
        results_bench.extend(s_total_bench)

        # --- Diffrax (Optimized) ---
        t0_diffrax = time.time()
        
        # 1. Stick
        D_stick = jnp.diag(jnp.array([0.0, 0.0, D_INTRA]))
        geom_stick = BlochTorreyGeometry(T1=T1, T2=T2, D=D_stick)
        
        sig_stick = simulate_pgse_shell(geom_stick, b_val, directions_jax, delta, Delta)
        # Block until ready to measure time accurately
        sig_stick.block_until_ready()
        
        # 2. Ball
        geom_ball = BlochTorreyGeometry(T1=T1, T2=T2, D=D_ISO)
        sig_ball = simulate_pgse_shell(geom_ball, b_val, directions_jax, delta, Delta)
        sig_ball.block_until_ready()
        
        s_total_diffrax = F_INTRA * sig_stick + (1 - F_INTRA) * sig_ball
        
        t1_diffrax = time.time()
        print(f"  Diffrax Time: {t1_diffrax - t0_diffrax:.5f} s")
        if b_val > 0:
            print(f"  Diffrax Speedup Factor vs Serial (Est): Expect >50x")
            
        results_diffrax.extend(s_total_diffrax)

    # Analysis
    res_bench = np.array(results_bench)
    res_diffrax = np.array(results_diffrax)
    
    # Normalize b=0
    b0_mask = np.concatenate([[True]*DIRECTIONS_PER_SHELL] + [[False]*DIRECTIONS_PER_SHELL*(len(B_VALUES)-1)])
    # Wait, simple mask construction based on loop logic
    # Just take first block
    s0_b = np.mean(res_bench[:DIRECTIONS_PER_SHELL])
    s0_d = np.mean(res_diffrax[:DIRECTIONS_PER_SHELL])
    
    res_bench /= s0_b
    res_diffrax /= s0_d
    
    mse = np.mean((res_bench - res_diffrax)**2)
    print(f"\nFinal MSE: {mse:.6e}")
    
    # Plotting
    # Flatten b-values
    b_flat = []
    for b in B_VALUES:
        b_flat.extend([b] * DIRECTIONS_PER_SHELL)
        
    plt.figure(figsize=(10,6))
    plt.scatter(b_flat, np.log(res_bench+1e-9), label='BENCH', alpha=0.5)
    plt.scatter(b_flat, np.log(res_diffrax+1e-9), label='Diffrax (Vmapped)', alpha=0.5, marker='x')
    plt.legend()
    plt.title(f"Optimized Validation (MSE={mse:.2e})")
    plt.savefig('bench_validation_optimized.png')
    print("Saved plot.")

if __name__ == "__main__":
    run_bench_validation()
