from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

class IsochromatPhantom(eqx.Module):
    """
    Represents a phantom composed of isochromats (independent spins).
    """
    positions: Float[Array, "N_spins 3"]
    T1: Float[Array, "N_spins"]
    T2: Float[Array, "N_spins"]
    M0: Float[Array, "N_spins"]
    off_resonance: Float[Array, "N_spins"]

    def __check_init__(self):
        """
        Validates the shapes of the phantom properties.
        """
        # Equinox modules are immutable, so we check attributes after initialization
        N_spins = self.positions.shape[0]
        if self.positions.shape != (N_spins, 3):
             raise ValueError(f"positions must have shape (N_spins, 3), got {self.positions.shape}")
        if self.T1.shape != (N_spins,):
            raise ValueError(f"T1 must have shape (N_spins,), got {self.T1.shape}")
        if self.T2.shape != (N_spins,):
             raise ValueError(f"T2 must have shape (N_spins,), got {self.T2.shape}")
        if self.M0.shape != (N_spins,):
             raise ValueError(f"M0 must have shape (N_spins,), got {self.M0.shape}")
        if self.off_resonance.shape != (N_spins,):
             raise ValueError(f"off_resonance must have shape (N_spins,), got {self.off_resonance.shape}")

class PulseSequence(eqx.Module):
    """
    Abstract base class for pulse sequences.
    """
    def get_gradients(self, t: Float[Array, ""]) -> Float[Array, "3"]:
        """
        Returns the gradient vector at time t.
        """
        raise NotImplementedError

    def get_rf(self, t: Float[Array, ""]) -> Complex[Array, ""]:
        """
        Returns the RF pulse amplitude (complex) at time t.
        """
        raise NotImplementedError

class TrapezoidalGradient(PulseSequence):
    """
    A simple trapezoidal gradient waveform.
    """
    amplitude: Float[Array, "3"]
    flat_time: Float[Array, ""]
    ramp_time: Float[Array, ""]
    start_time: Float[Array, ""]

    def get_gradients(self, t: Float[Array, ""]) -> Float[Array, "3"]:
        """
        Calculates the gradient at time t for a trapezoidal shape.
        """
        # Calculate relative time
        dt = t - self.start_time
        
        # Ramp up
        ramp_up = jnp.clip(dt / self.ramp_time, 0.0, 1.0)
        
        # Ramp down
        # Time when ramp down starts: ramp_time + flat_time
        # Time when ramp down ends: 2*ramp_time + flat_time
        # We want 1.0 at start of ramp down, 0.0 at end
        
        # Let's define it piecewise for clarity, then optimize/vectorize if needed
        # But for JAX compatibility, we should use jnp.where or similar logic
        
        # Total duration
        total_duration = 2 * self.ramp_time + self.flat_time
        
        # Check if we are within the pulse duration
        is_active = (dt >= 0) & (dt <= total_duration)
        
        # Calculate normalized amplitude profile
        # Ramp up: t < ramp_time -> t / ramp_time
        # Flat: ramp_time <= t < ramp_time + flat_time -> 1.0
        # Ramp down: ramp_time + flat_time <= t < total_duration -> 1.0 - (t - (ramp_time + flat_time)) / ramp_time
        
        profile_up = dt / self.ramp_time
        profile_flat = 1.0
        profile_down = 1.0 - (dt - (self.ramp_time + self.flat_time)) / self.ramp_time
        
        profile = jnp.where(dt < self.ramp_time, profile_up,
                            jnp.where(dt < self.ramp_time + self.flat_time, profile_flat,
                                      jnp.where(dt <= total_duration, profile_down, 0.0)))
        
        return jnp.where(is_active, self.amplitude * profile, jnp.zeros(3))

    def get_rf(self, t: Float[Array, ""]) -> Complex[Array, ""]:
        """
        No RF for a gradient object.
        """
        return jnp.array(0.0, dtype=jnp.complex64)
