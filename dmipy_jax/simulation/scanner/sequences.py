
import jax.numpy as jnp
from typing import NamedTuple, Optional, Union

class SpinEchoSequence(NamedTuple):
    """
    Represents a standard Spin Echo sequence for simulation.
    
    This is a simplified representation commonly used in analytical models 
    and fast benchmarks.
    """
    TE: float  # Echo Time (s)
    TR: float  # Repetition Time (s)
    gradients: jnp.ndarray  # (N_steps, 3) - Effective gradients or waveform moments
    
class GeneralSequence(NamedTuple):
    """
    Represents a general sequence defined by arbitrary waveforms.
    
    This is the target for PyPulseq conversion where full waveform fidelity is needed.
    """
    time_points: jnp.ndarray # (N_t,) Time points of the simulation grid
    gradients: jnp.ndarray   # (N_t, 3) Gradient waveform [T/m]
    rf_amplitude: jnp.ndarray # (N_t,) RF amplitude [T] (complex or magnitude + phase?)
                              # Typically strictly complex for Bloch.
    rf_phase: jnp.ndarray    # (N_t,) RF phase [rad]
    adc_mask: jnp.ndarray    # (N_t,) 1 if ADC is on, 0 otherwise
