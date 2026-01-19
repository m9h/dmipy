import sys
import os
import jax
import numpy as np
import pypulseq as pp

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dmipy_jax.simulation.scanner.phantoms import SyntheticBigMacPhantom
from dmipy_jax.external.pulseq import pulseq_to_general
from dmipy_jax.simulation.scanner.bloch import simulate_acquisition

def main():
    print("=== JAX-POSSUM Demo: PyPulseSeq + BigMac ===")
    
    # 1. Create a PyPulseSeq Sequence
    # Simple Diffusion Weighted Spin Echo
    print("Generating PyPulseSeq sequence...")
    system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s', 
                     rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    
    seq = pp.Sequence(system)
    
    # Define parameters (TE=80ms, b=1000)
    # Simplified for demo speed: simpler gradient
    # A simple PGSE
    
    # RF
    rf90 = pp.make_sinc_pulse(flip_angle=90 * np.pi/180, duration=2e-3, system=system, 
                             slice_thickness=5e-3, apodization=0.5, time_bw_product=4)
    rf180 = pp.make_block_pulse(flip_angle=180 * np.pi/180, duration=2e-3, system=system)
    
    # Gradients
    # b=1000 s/mm^2 = 1e9 s/m^2
    # G = 40 mT/m
    # delta = 20 ms
    # DELTA = 40 ms
    # gamma = 267.5e6 rad/s/T
    # b ~ (gamma G delta)^2 (DELTA - delta/3)
    
    # For speed, let's just do a dummy gradient pulse to verify the pipeline
    g = pp.make_trapezoid(channel='x', amplitude=40e-3, duration=10e-3, system=system) # 40 mT/m, 10ms
    
    # Sequence blocks
    seq.add_block(rf90)
    seq.add_block(pp.make_delay(0.01))
    seq.add_block(g)
    seq.add_block(pp.make_delay(0.01))
    seq.add_block(rf180)
    seq.add_block(pp.make_delay(0.01))
    seq.add_block(g)
    seq.add_block(pp.make_delay(0.01))
    
    # ADC
    adc = pp.make_adc(num_samples=256, duration=5e-3, system=system, delay=0.005)
    seq.add_block(adc)
    
    # 2. Convert to JAX-POSSUM format
    print("Converting to JAX-POSSUM GeneralSequence (rasterizing)...")
    # Using coarse raster for demo speed
    dt = 100e-6 # 100us
    jax_seq = pulseq_to_general(seq, dt=dt)
    
    print(f"Sequence Duration: {jax_seq.time_points[-1]:.4f} s")
    print(f"Time steps: {jax_seq.time_points.shape[0]}")
    
    # 3. Instantiate BigMac Phantom
    print("Creating BigMac Phantom...")
    # Low spin count for rapid testing
    phantom = SyntheticBigMacPhantom(n_spins=1000, radius=0.05, seed=1337)
    print(f"Phantom created with {phantom.positions.shape[0]} spins.")
    
    # 4. Simulate
    print("Running Bloch Simulation on GPU (if available)...")
    duration = jax_seq.time_points[-1]
    
    import time
    t0 = time.time()
    
    signal = simulate_acquisition(phantom, jax_seq, duration=duration)
    # Force computation
    signal.block_until_ready()
    
    t1 = time.time()
    print(f"Simulation completed in {t1-t0:.2f} seconds.")
    print(f"Resulting Signal (Complex Sum): {signal}")
    print(f"Magnitude: {float(jax.numpy.abs(signal))}")
    print(f"Phase: {float(jax.numpy.angle(signal))}")

if __name__ == "__main__":
    main()
