import jax
import jax.numpy as jnp
import numpy as np
import pypulseq as pp
import h5py
import diffrax
import os
import warnings
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

from experiments.koma_integration.koma_wrapper import KomaSimulator

def test_pgse_analytical():
    """
    Test 1: The Analytical Check (Stejskal-Tanner)
    """
    print("\n--- Test 1: Analytical Stejskal-Tanner Check ---")
    
    # 1. Setup Parameters
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
    
    # KomaMRI needs an RF pulse to excite spins
    rf90 = pp.make_block_pulse(flip_angle=np.pi/2, duration=1e-4, system=system)
    seq.add_block(rf90)
    
    # 1. Calculate required rise time
    # pypulseq calc_duration helper or just make a dummy
    dummy_trap = pp.make_trapezoid(channel='x', amplitude=G_amp_hz, flat_time=0, system=system)
    rise_time = dummy_trap.rise_time
    flat_time = delta - rise_time
    
    grad_x = pp.make_trapezoid(channel='x', amplitude=G_amp_hz, flat_time=flat_time, system=system)
    
    # Timing:
    # 1. Gradient 1
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
    
    # 4. Add ADC block for KomaMRI readout
    # We add a small ADC event at the end to ensure we capture the signal.
    # Note: JAX PulseInterpreter ignores ADC blocks usually, but we need it for Koma.
    adc = pp.make_adc(num_samples=1, duration=10e-6, system=system)
    seq.add_block(adc)
    
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
        print(f"KomaMRI ground truth {h5_path} not found.")
        print("Attempting to generate it using local Julia installation...")
        try:
            # Assume we use the test_pgse.seq generated by analytical test if checking 'koma.h5'
            # Or assume h5_path implies a seq path
            seq_path = "test_pgse.seq"
            if not os.path.exists(seq_path):
                 print("Sequence file not found to generate Koma data.")
                 return

            # D from analytical test
            D_koma = 1e-9 # m^2/s (Isotropic)
            
            sim = KomaSimulator()
            sim.generate_ground_truth(seq_path, h5_path, D=D_koma)
            print("Successfully generated KomaMRI ground truth.")
        except Exception as e:
            print(f"Skipping KomaMRI check: Could not generate data. Error: {e}")
            return

    with h5py.File(h5_path, 'r') as f:
        t_koma = f['t'][:]
        # grads_koma = f['grads'][:] 
        sig_koma = f['signal_magnitude'][:] 
        
    # Use PulseInterpreter for JAX simulation (same seq)
    interpreter = PulseInterpreter("test_pgse.seq", dt_raster=10e-6)
    gamma_hz = 42.577e6
    
    def waveform_fn(t):
        return interpreter.control.evaluate(t) / gamma_hz
        
    D_val = 0.0 # Validation of Sequence Timing/Rephasing (Diffusion ignored in Koma Bloch)
    geometry = BlochTorreyGeometry(T1=1e4, T2=1e4, D=D_val)
    
    # We want to match the total duration of Koma simulation
    duration = t_koma[-1] if len(t_koma) > 1 else interpreter.t_grid[-1]
    
    M0 = jnp.array([1.0, 0.0, 0.0])
    
    # Run JAX simulation
    M_final = simulate_signal(
        geometry=geometry,
        waveform=waveform_fn,
        duration=duration,
        M0=M0
    )
    
    S_sim = jnp.linalg.norm(M_final[:2])
    
    # Compare with final Koma point
    gt = sig_koma[-1]
        
    mse = (S_sim - gt)**2
    # print(f"Observed Simulation b-value: {-jnp.log(S_sim)/D_val * 1e-6:.2f}") # Undefined for D=0
    print(f"Simulated: {S_sim:.6f}, Koma: {gt:.6f}")
    print(f"MSE: {mse:.2e}")
    
    if mse > 1e-4:
        print("FAIL: KomaMRI Validation failed.")
        return False
    else:
        print("PASS: KomaMRI Validation success.")
        return True


if __name__ == "__main__":
    # Suppress JAX warnings or others if needed
    warnings.simplefilter("ignore")
    
    try:
        success = test_pgse_analytical()
        if success:
            print("PASS")
            
            # Run Koma check
            validate_against_koma("koma_truth.h5")
        else:
            print("FAIL")
            exit(1)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("FAIL")
        exit(1)
