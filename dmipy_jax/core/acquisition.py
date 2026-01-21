import jax.numpy as jnp
import numpy as np
from typing import Optional, Any
import equinox as eqx

class SimpleAcquisitionScheme(eqx.Module):
    """
    A simplified AcquisitionScheme for dmipy-jax that avoids legacy dependency issues.
    """
    bvalues: Any
    gradient_directions: Any
    b0_threshold: float
    b0_mask: Any
    delta: Optional[Any]
    Delta: Optional[Any]
    TE: Optional[Any]
    TR: Optional[Any]
    qvalues: Optional[Any]
    gradient_strengths: Optional[Any]
    shell_indices: Optional[Any] = None
    unique_shell_indices: Optional[Any] = None

    def __init__(self, bvalues, gradient_directions, 
                 delta=None, Delta=None, TE=None, TR=None,
                 qvalues=None, gradient_strengths=None,
                 b0_threshold=10e6):
        
        self.bvalues = jnp.array(bvalues)
        print(f"SimpleAcquisitionScheme init: bvalues shape {self.bvalues.shape}")
        self.gradient_directions = jnp.array(gradient_directions)
        self.b0_threshold = b0_threshold
        self.b0_mask = self.bvalues <= b0_threshold
        
        # Optional parameters
        self.delta = jnp.array(delta) if delta is not None else None
        self.Delta = jnp.array(Delta) if Delta is not None else None
        self.TE = jnp.array(TE) if TE is not None else None
        self.TR = jnp.array(TR) if TR is not None else None
        
        # Calculated if not provided
        self.qvalues = qvalues
        self.gradient_strengths = gradient_strengths
        
        # Helper for shell indices (simplified)
        self.shell_indices = None
        self.unique_shell_indices = None
        # We can implement basic shell clustering if needed, but for now we leave it simple.
        
    @property
    def number_of_measurements(self):
        return len(self.bvalues)
        
    def print_acquisition_info(self):
        print(f"SimpleAcquisitionScheme: {self.number_of_measurements} measurements")
        print(f"b-values range: {jnp.min(self.bvalues)} - {jnp.max(self.bvalues)}")

def acquisition_scheme_from_bvalues(bvalues, gradient_directions, 
                                    delta=None, Delta=None, TE=None, TR=None,
                                    min_b_shell_distance=50e6, b0_threshold=10e6):
    """
    Factory function matching dmipy signature but returning SimpleAcquisitionScheme.
    """
    # Simple tiling logic if scalars provided
    n = len(bvalues)
    
    def _tile_if_scalar(x, n):
        if x is None: return None
        if np.isscalar(x) or (isinstance(x, (list, tuple)) and len(x)==1):
             # Use numpy for tiling to match expected behavior before JAX conversion
             return np.full(n, x)
        return x

    delta_ = _tile_if_scalar(delta, n)
    Delta_ = _tile_if_scalar(Delta, n)
    TE_ = _tile_if_scalar(TE, n)
    TR_ = _tile_if_scalar(TR, n)
    
    return SimpleAcquisitionScheme(bvalues, gradient_directions, 
                                   delta=delta_, Delta=Delta_, TE=TE_, TR=TR_,
                                   b0_threshold=b0_threshold)
