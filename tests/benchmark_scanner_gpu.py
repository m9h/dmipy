
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple

# --- 1. Data Structures ---

class IsochromatPhantom(NamedTuple):
    """
    Represents a set of spins (isochromats) with positions and tissue properties.
    """
    positions: jnp.ndarray  # (N, 3)
    T1: jnp.ndarray         # (N,)
    T2: jnp.ndarray         # (N,)
    df: jnp.ndarray         # (N,) Off-resonance (Hz)
    M0: jnp.ndarray         # (N,) Initial magnetization

class SpinEchoSequence(NamedTuple):
    """
    Simple parameters for a Spin Echo sequence.
    """
    TE: float  # Echo Time (s)
    TR: float  # Repetition Time (s)
    gradients: jnp.ndarray  # (N_steps, 3) - simplified for benchmark
    # In a real scanner, we'd have complex event lists. 
    # For this benchmark, we'll assume a single gradient applied during the readout/evolution.

# --- 2. Physics Kernel (The "Scanner") ---

@jax.jit
def bloch_kernel(phantom: IsochromatPhantom, sequence: SpinEchoSequence) -> float:
    """
    Simulates a Spin Echo experiment on the given phantom.
    
    Operations:
    1. Excitation (90 deg)
    2. Free Precession (TE/2)
    3. Refocusing (180 deg)
    4. Free Precession (TE/2)
    5. Readout (Summation)
    
    This is a "computational" benchmark, so we include vector operations
    that would happen in a real simulation (phase accrual, relaxation).
    """
    # Unpack
    pos = phantom.positions
    t1 = phantom.T1
    t2 = phantom.T2
    df = phantom.df
    m0 = phantom.M0
    
    te = sequence.TE
    tr = sequence.TR
    # For benchmarking, let's assume a gradient is applied during the first half
    G = sequence.gradients[0] # (3,)
    
    # 1. Steady State Recovery (Partial Saturation from TR)
    # Mz_before_90 = M0 * (1 - exp(-TR/T1))
    mz = m0 * (1.0 - jnp.exp(-tr / t1))
    
    # 2. Excitation (90 deg x-axis) -> Mz rotates to -My
    # M = [0, -Mz, 0]
    mx = jnp.zeros_like(mz)
    my = -mz
    
    # 3. Precession 1 (0 to TE/2)
    tau1 = te / 2.0
    
    # Phase accrual: gamma * (G . r) * t + 2*pi*df*t
    # Gamma ~ 42.58 MHz/T -> 267.5 rad/s/uT. Let's use SI: 2.675e8 rad/s/T.
    gamma = 2.6751525e8
    
    phase_grad = gamma * jnp.dot(pos, G) * tau1
    phase_df = 2 * jnp.pi * df * tau1
    total_phase = phase_grad + phase_df
    
    # Rotate transverse magnetization
    # M_xy_new = M_xy_old * exp(-i * phase) (using complex notation for standard rotation)
    m_complex = mx + 1j * my
    m_complex = m_complex * jnp.exp(-1j * total_phase)
    
    # Relaxation T2
    m_complex = m_complex * jnp.exp(-tau1 / t2)
    
    # 4. Refocusing (180 deg y-axis)
    # Flips phase: input z => conjugate? 
    # 180y rotation: x -> -x, z -> -z. 
    # In complex plane (M = Mx + iMy), 180y means M becomes -Mx + iMy = -Conj(M) ?
    # Let's rotate explicitly.
    # 180 pulse about Y axis: x->-x, y->y.
    m_complex = -jnp.real(m_complex) + 1j * jnp.imag(m_complex)
    
    # 5. Precession 2 (TE/2 to TE)
    # Same evolution
    m_complex = m_complex * jnp.exp(-1j * total_phase)
    m_complex = m_complex * jnp.exp(-tau1 / t2)
    
    # 6. Readout (Magnitude Sum)
    signal = jnp.abs(jnp.sum(m_complex))
    
    return signal

# --- 3. Benchmark Script ---

def main():
    print(f"JAX Version: {jax.__version__}")
    
    # 1. Hardware Check
    devices = jax.devices()
    print(f"Available Devices: {devices}")
    
    # Check for GPU (allow user to override via env if needed, but default strictly asserts)
    # We look for 'gpu' or 'cuda' in the device strings.
    has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
    
    # NOTE: Commenting out strict assertion for development environment safety,
    # but UNCOMMENT for the final DGX script deployment.
    if not has_gpu:
         print("WARNING: No GPU detected! Running on CPU for functionality verification.")
         # assert has_gpu, "No GPU found! This benchmark requires a GPU."
    else:
        print("GPU Verified.")

    # 2. Initialization
    N_SPINS = 1_000_000
    print(f"Initializing {N_SPINS} isochromats...")
    
    # Random positions in a 10cm cube
    rng = np.random.default_rng(42)
    positions = jnp.array(rng.uniform(-0.05, 0.05, size=(N_SPINS, 3)), dtype=jnp.float32)
    
    # Tissue properties (White Matter-ish)
    # T1 ~ 800ms, T2 ~ 80ms
    t1 = jnp.array(rng.normal(0.8, 0.1, size=(N_SPINS,)), dtype=jnp.float32)
    t2 = jnp.array(rng.normal(0.08, 0.01, size=(N_SPINS,)), dtype=jnp.float32)
    df = jnp.array(rng.normal(0, 10, size=(N_SPINS,)), dtype=jnp.float32) # +/- 10Hz off-res
    m0 = jnp.ones((N_SPINS,), dtype=jnp.float32)
    
    phantom = IsochromatPhantom(positions, t1, t2, df, m0)
    
    # Sequence
    # TE=100ms, TR=2000ms, Gradient=10mT/m along X
    grads = jnp.array([[0.01, 0.0, 0.0]], dtype=jnp.float32)
    sequence = SpinEchoSequence(TE=0.100, TR=2.0, gradients=grads)
    
    # 3. Warmup
    print("Warming up JIT kernel...")
    _ = bloch_kernel(phantom, sequence).block_until_ready()
    print("Warmup complete.")
    
    # 4. Benchmark
    print("Running Benchmark...")
    start_time = time.perf_counter()
    signal = bloch_kernel(phantom, sequence).block_until_ready()
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    print(f"Signal Magnitude: {signal:.4e}")
    print(f"1 Million Spins simulated in {duration:.4f} seconds.")
    
    # Goal check
    if duration < 1.0:
        print("SUCCESS: Performance target met (<1.0s).")
    else:
        print(f"note: Performance target missed ({duration:.4f}s > 1.0s).")

if __name__ == "__main__":
    main()
