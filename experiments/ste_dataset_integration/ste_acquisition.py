import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from typing import Optional, Any
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

class STEAcquisitionScheme(SimpleAcquisitionScheme):
    """
    Stimulated Echo (STE) Acquisition Scheme.
    
    Models the signal decay distinct to STE sequences, specifically accounting for
    T1 relaxation during the mixing time (TM) and T2 relaxation during TE/2.
    
    Signal ~ (1/2) * exp(-TM/T1) * exp(-TE/T2)
    
    Attributes:
        mixing_time (Float[Array, "N"]): The mixing time (TM) in seconds.
    """
    mixing_time: Float[Array, "N"]
    
    def __init__(
        self, 
        bvalues: Any, 
        gradient_directions: Any, 
        mixing_time: Any,
        delta: Optional[Any] = None, 
        TE: Optional[Any] = None, 
        TR: Optional[Any] = None,
        b0_threshold: float = 10e6
    ):
        """
        Initialize the STE Acquisition Scheme.
        
        Args:
            bvalues: b-values in s/m^2.
            gradient_directions: (N, 3) gradient vectors.
            mixing_time: Mixing time (TM) in seconds.
            delta: Pulse duration in seconds.
            TE: Echo time in seconds.
            TR: Repetition time in seconds.
            b0_threshold: Threshold for b0 identification.
        """
        # Broadcast to match bvalues length if necessary
        n_measurements = len(bvalues)
        
        # Helper to broadcast scalars/0-d arrays to 1-d
        def _ensure_shape(arr, n):
            arr = jnp.array(arr)
            if arr.ndim == 0:
                return jnp.full((n,), arr)
            return arr
            
        mixing_time_arr = _ensure_shape(mixing_time, n_measurements)
        
        # Physics constraint: Calculate Delta (Diffusion Time)
        Delta_arr = None
        if delta is not None:
             delta_arr = _ensure_shape(delta, n_measurements)
             Delta_arr = mixing_time_arr + delta_arr / 3.0
        
        # Also ensure delta is broadcasted for super call if it was provided
        delta_arg = _ensure_shape(delta, n_measurements) if delta is not None else None
             
        super().__init__(
            bvalues=bvalues,
            gradient_directions=gradient_directions,
            delta=delta_arg,
            Delta=Delta_arr,
            TE=TE,
            TR=TR,
            b0_threshold=b0_threshold
        )
        
        self.mixing_time = mixing_time_arr

    @property
    def diffusion_time(self):
        """Alias for Delta."""
        return self.Delta

    def print_acquisition_info(self):
        super().print_acquisition_info()
        print(f"Mixing Time (TM) range: {jnp.min(self.mixing_time):.4f} - {jnp.max(self.mixing_time):.4f} s")

if __name__ == "__main__":
    # Self-test
    print("Running STEAcquisitionScheme Self-Test...")
    bvals = jnp.array([0., 1000., 2000., 3000.])
    bvecs = jnp.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tm = 0.5 # 500ms
    delta = 0.01 # 10ms
    
    scheme = STEAcquisitionScheme(bvals, bvecs, mixing_time=tm, delta=delta)
    scheme.print_acquisition_info()
    
    expected_delta = tm + delta / 3.0
    print(f"Calculated Delta: {scheme.Delta[0]}")
    print(f"Expected Delta: {expected_delta}")
    assert jnp.allclose(scheme.Delta, expected_delta)
    print("Test Passed.")
