
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Complex
from typing import NamedTuple, Optional, Union

class SpinEchoSequence(eqx.Module):
    """
    Represents a standard Spin Echo sequence for simulation.
    """
    TE: float
    TR: float
    gradients: Array

    def get_gradients(self, t):
        # Simplified: return 0 or effective gradient? 
        # For now return 0 as placeholder or constant
        return jnp.zeros(3)
        
    def get_rf(self, t):
        return 0.0j

class GeneralSequence(eqx.Module):
    """
    Represents a general sequence defined by arbitrary waveforms.
    """
    time_points: Array
    gradients: Array
    rf_amplitude: Array
    rf_phase: Array
    adc_mask: Array

    def get_gradients(self, t):
        # Interpolate gradients at time t
        # gradients: (N_t, 3)
        # t can be scalar or array (if vmapped over t? No, t is scalar in ODE)
        
        # Use jnp.interp
        # We need to interp for each channel
        # Use vmap internally? Or just explicit
        
        g_x = jnp.interp(t, self.time_points, self.gradients[:, 0], left=0., right=0.)
        g_y = jnp.interp(t, self.time_points, self.gradients[:, 1], left=0., right=0.)
        g_z = jnp.interp(t, self.time_points, self.gradients[:, 2], left=0., right=0.)
        
        return jnp.stack([g_x, g_y, g_z])

    def get_rf(self, t):
        # Interpolate RF
        amp = jnp.interp(t, self.time_points, self.rf_amplitude, left=0., right=0.)
        phase = jnp.interp(t, self.time_points, self.rf_phase, left=0., right=0.)
        return amp * jnp.exp(1j * phase)
