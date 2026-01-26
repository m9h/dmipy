import jax
import jax.numpy as jnp
import numpy as np
import pypulseq as pp
import h5py
import diffrax
import os
from typing import Optional

# Adjust imports based on project structure
try:
    from dmipy_jax.pulseq.pulse_interpreter import PulseInterpreter
    from dmipy_jax.simulation.scanner.bloch import simulate_signal, BlochTorreyGeometry
except ImportError:
    # Fallback if running from root without package installed in editable mode
    import sys
    sys.path.append(os.path.abspath("."))
    from dmipy_jax.pulseq.pulse_interpreter import PulseInterpreter
    from dmipy_jax.simulation.scanner.bloch import simulate_signal, BlochTorreyGeometry

def test_pgse_analytical():
    """
    Test 1: The Analytical Check (Stejskal-Tanner)
    """
    print("\n--- Test 1: Analytical Stejskal-Tanner Check ---")
    
    # 1. Setup Parameters
    # We want to match b = (gamma * G * delta)^2 * (Delta - delta/3)
    # This requires Rectangular pulses.
    # We will approximate this with very fast ramps (high slew rate).
    
    G_amp = 30e-3 # 30 mT/m
    delta = 10e-3 # 10 ms
    Delta = 40e-3 # 40 ms
    
    # Constants
    # Gyromagnetic ratio for protons
    gamma = 2.6751525e8 # rad/s/T
    # Start with D = 1e-3 mm^2/s = 1e-9 m^2/s
    D_val_mm2s = 1e-3
    D_val_si = D_val_mm2s * 1e-6 # m^2/s
    
    # 2. Sequence Construction (PyPulseq)
    # High slew to approximate Rectangle
    system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s')
    seq = pp.Sequence(system)
    
    # Create gradient blocks
    # Note: pypulseq expects amplitude in Hz/m usually!
    gamma_hz = 42.577e6 # Hz/T
    G_amp_hz = G_amp * gamma_hz
    
    # We want flat time to be 'delta'. 
    
    # Rise time for 30mT/m at 200T/m/s is 0.15 ms.
    # To match 'delta' in analytical formula (rectangular), we want the Area to match.
    # Area Rect = G * delta
    # Area Trap = G * (flat + rise) (since fall=rise)
    # So flat + rise = delta  =>  flat = delta - rise
    
    # 1. Calculate required rise time
    # pypulseq calc_duration helper or just make a dummy
    dummy_trap = pp.make_trapezoid(channel='x', amplitude=G_amp_hz, flat_time=0, system=system)
    rise_time = dummy_trap.rise_time
    flat_time = delta - rise_time
    
    grad_x = pp.make_trapezoid(channel='x', amplitude=G_amp_hz, flat_time=flat_time, system=system)
    
    # Timing:
    # 1. Gradient 1
    # 2. 180 Pulse at Delta/2 ?
    # Standard PGSE:
    # 90 -> ... -> G1 -> ... -> 180 -> ... -> G2 -> ... -> Echo
    # Separation between START of G1 and START of G2 is Delta?
    # Or Center to Center?
    # Stejskal-Tanner Delta is time between onset of pulses (if rectangular) or centers.
    # Let's assume Center-to-Center separation is Delta.
    
    # Build sequence
    # 1. Grad 1
    seq.add_block(grad_x)
    
    # 2. Delay to next Grad
    # Time from end of G1 to start of G2.
    # Center 1 is at (ramp + flat + ramp)/2 = duration/2.
    # Center 2 should be at Center 1 + Delta.
    # So Start 2 should be at Start 1 + Delta.
    # Wait = Delta - duration.
    # Duration = rise + flat + fall
    block_duration = grad_x.rise_time + grad_x.flat_time + grad_x.fall_time
    delay_time = Delta - block_duration
    
    if delay_time < 0:
        raise ValueError("Delta must be > gradient duration")
        
    seq.add_block(pp.make_delay(delay_time))
    
    # 3. Grad 2 (Effective -G if we don't have simulated 180refoc)
    grad_x_neg = pp.make_trapezoid(channel='x', amplitude=-G_amp_hz, flat_time=flat_time, system=system)
    seq.add_block(grad_x_neg)
    
    # Save seq
    seq_path = "test_pgse.seq"
    seq.write(seq_path)
    
    # 3. Interpreter
    interpreter = PulseInterpreter(seq_path, dt_raster=10e-6)
    duration = interpreter.t_grid[-1]
    
    # Calculate Theoretical b-value (Rectangular Approx)
    b_rect = (gamma * G_amp * delta)**2 * (Delta - delta / 3.0)
    print(f"Rectangular Approx b-value: {b_rect * 1e-6:.2f} s/mm^2")

    # Calculate Exact b-value from Waveform (Trapezoidal)
    # We numerically integrate the sequence we just built.
    print("Computing exact b-value from waveform integration...")
    dt_calc = 1e-6 # Finer grid for integration
    t_calc = jnp.arange(0, duration, dt_calc)
    
    # helper to get G in T/m
    def get_g_tesla(t_arr):
        # vmap evaluation
        # interpreter.control.evaluate is vectorised?
        # LinearInterpolation evaluate is vectorised if t is array?
        # Usually yes in JAX.
        return jax.vmap(interpreter.control.evaluate)(t_arr) / gamma_hz

    G_t = get_g_tesla(t_calc) # Shape (N, 3)
    
    # k(t) = gamma * integral(G)
    # cumsum
    k_t = jnp.cumsum(G_t, axis=0) * dt_calc * gamma # (N, 3)
    
    # b = integral(k^2)
    k_sq = jnp.sum(k_t**2, axis=1) # (N,)
    b_exact = jnp.sum(k_sq) * dt_calc
    
    print(f"Exact Waveform b-value: {b_exact * 1e-6:.2f} s/mm^2")
    
    signal_theory = np.exp(-b_exact * D_val_si)
    print(f"Theoretical Signal (Exact): {signal_theory:.6f}")

    # 5. Simulation
    # Define Geometry
    # T1, T2 infinite to isolate diffusion effect.
    geometry = BlochTorreyGeometry(T1=1e6, T2=1e6, D=D_val_si)
    
    # Wrapper function for waveform
    def waveform_fn(t):
        return interpreter.control.evaluate(t) # Returns [Gx, Gy, Gz] in Hz/m?
        # WAIT. PulseInterpreter says it returns Hz/m?
        # But simulate_signal doc says "f(t) -> [Gx, Gy, Gz] (T/m)".
        # Let's check PulseInterpreter again.
        # "We store as is (Hz/m) and let the BlochTerm handle Gamma."
        # BlochSimulator.py: "omega_z = 2 * jnp.pi * f_hz ... ignore self.gamma because input ALREADY in Hz"
        # BUT `simulation/scanner/bloch.py` says:
        # "dk_dt = GYRO_MAGNETIC_RATIO * G"
        # "G = waveform(t)"
        # This implies `bloch.py` expects G in Tesla/m.
        # PulseInterpreter returns Hz/m (gamma * G).
        # We need to convert back to T/m if we use `bloch.py`'s `simulate_signal` directly!
        # Or modify how we call it.
        # Hz/m / (gamma_hz) ? 
        # Pulseq: Hz/m = gamma_bar * G (gamma_bar = 42.57 MHz/T).
        # bloch.py uses GYRO_MAGNETIC_RATIO which is likely rad/s/T (2.67e8).
        # We need to be careful with 2pi.
        # PulseInterpreter:
        # "Pulseq defines gradients in Hz/m (gamma * G)." (Usually gamma_bar?)
        # "We store as is (Hz/m)"
        # If PulseInterpreter returns G_hz_per_m, and bloch.py computes:
        # dk_dt = gamma * G
        # If we pass G_hz_per_m, we get dk_dt = gamma * G_hz_per_m -> Double gamma?
        
        # FIX: We must convert PulseInterpreter output to T/m for `simulate_signal`.
        # G_TeslaPerMeter = G_HzPerMeter / (gamma_bar_HzPerTesla)
        # gamma_bar = 42.57e6 Hz/T.
        pass
    
    # Re-reading PulseInterpreter:
    # "Pulseq defines gradients in Hz/m (gamma * G). "
    # "The Bloch solver usually expects T/m or simple G/cm."
    
    # We need to bridge this.
    gamma_hz = 42.577e6 # Hz/T
    
    
    def waveform_t_per_m(t):
        g_hz = interpreter.control.evaluate(t)
        return g_hz / gamma_hz
        
    duration = interpreter.t_grid[-1]
    
    # Run Simulation
    # Start with M0 along X to simulate after-90-degree pulse
    M0 = jnp.array([1.0, 0.0, 0.0])
    
    M_final = simulate_signal(
        geometry=geometry,
        waveform=waveform_t_per_m,
        duration=duration,
        M0=M0,
        dt=1e-5 # Fine step
    )
    
    # Final Signal
    # M_final is [Mx, My, Mz]
    # We want transverse magnitude
    S_sim = jnp.linalg.norm(M_final[:2])
    
    b_observed = -jnp.log(S_sim) / D_val_si
    print(f"Observed Simulation b-value: {b_observed * 1e-6:.2f} s/mm^2")
    
    print(f"Simulated Signal: {S_sim:.6f}")
    
    diff = jnp.abs(S_sim - signal_theory)
    print(f"Difference: {diff:.2e}")
    
    if diff > 1e-3:
        print("FAIL: Analytical Check failed.")
        # Debug info
        print(f"  G_amp: {G_amp} T/m")
        print(f"  D: {D_val_si} m^2/s")
        print(f"  b_rect: {b_rect}")
        print(f"  b_exact: {b_exact}")
        return False
    else:
        print("PASS: Analytical Check success.")
        return True

def validate_against_koma(h5_path):
    """
    Test 2: The KomaMRI Check (External Validation)
    """
    print("\n--- Test 2: KomaMRI Validation ---")
    
    if not os.path.exists(h5_path):
        print(f"Skipping: File {h5_path} not found.")
        return

    with h5py.File(h5_path, 'r') as f:
        t_koma = f['t'][:]
        grads_koma = f['grads'][:] # Shape (N, 3) presumably T/m or Hz/m?
        # Assume Koma stores T/m usually, or we check attributes.
        # Let's assume T/m for now based on prompt saying "Run the JAX simulator using the loaded gradients".
        
        sig_koma = f['signal_magnitude'][:] # Ground truth scalar or (N_t,)? 
        # Usually checking endpoint signal?
        # Or timecourse? "Print MSE between JAX prediction and KomaMRI ground truth".
        # If it's a single value (final signal), MSE is just squared diff.
        
    # Setup Interpolator
    t_jax = jnp.array(t_koma)
    g_jax = jnp.array(grads_koma)
    
    interp = diffrax.LinearInterpolation(ts=t_jax, ys=g_jax)
    
    # Geometry? 
    # We need to know what D was used in Koma!
    # Assume it's stored in h5 or fixed.
    # For validation, we might assume specific params or read from attributes.
    # Placeholder: D = 1e-9 m^2/s, T1/T2 infinity.
    D_val = 1e-9 # Standard free water-ish
    geometry = BlochTorreyGeometry(T1=10.0, T2=10.0, D=D_val)
    
    def waveform_fn(t):
        return interp.evaluate(t)
        
    duration = t_koma[-1]
    M0 = jnp.array([1.0, 0.0, 0.0])
    
    M_final = simulate_signal(
        geometry=geometry,
        waveform=waveform_fn,
        duration=duration,
        M0=M0
    )
    
    S_sim = jnp.linalg.norm(M_final[:2])
    
    # Compare
    # Assuming sig_koma is the final value
    gt = sig_koma
    if isinstance(gt, np.ndarray) and gt.size > 1:
        # If it's a timecourse, we can't easily compare unless we simulate timecourse.
        # Assume it's final value.
        gt = gt[-1]
        
    mse = (S_sim - gt)**2
    print(f"Simulated: {S_sim}, Koma: {gt}")
    print(f"MSE: {mse:.2e}")

import warnings
if __name__ == "__main__":
    # Suppress JAX warnings or others if needed
    warnings.simplefilter("ignore")
    
    try:
        success = test_pgse_analytical()
        if success:
            print("PASS")
        else:
            print("FAIL")
            exit(1)
            
        # Optional: Run Koma check if file exists
        # validate_against_koma("koma_truth.h5")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("FAIL")
        exit(1)
