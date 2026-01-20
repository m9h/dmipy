import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
import numpy as np

@dataclass
class JaxAcquisition:
    """
    Simplified acquisition scheme configuration for JAX-based models.

    This class strips away the complexity of managing shells, unique pulse timings,
    and string formatting/plotting found in the original DmipyAcquisitionScheme.
    It focuses on providing the essential data arrays (b-values, gradient vectors,
    timings) as JAX arrays, ready for accelerator computation.

    Args:
        bvalues (Union[jnp.ndarray, float]): The b-values (diffusion weighting factors)
            in s/m^2.
        gradient_directions (jnp.ndarray): The unit gradient direction vectors.
            Shape: (N_measurements, 3).
        delta (Optional[Union[jnp.ndarray, float]]): The pulse duration (small delta) in seconds.
        Delta (Optional[Union[jnp.ndarray, float]]): The pulse separation (big Delta) in seconds.
    """
    bvalues: Union[jnp.ndarray, float]
    gradient_directions: jnp.ndarray
    delta: Optional[Union[jnp.ndarray, float]] = None
    Delta: Optional[Union[jnp.ndarray, float]] = None
    echo_time: Optional[Union[jnp.ndarray, float]] = None
    total_readout_time: Optional[Union[jnp.ndarray, float]] = None
    btensors: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """
        Immediately convert all inputs to JAX arrays to ensure compatibility
        with JAX transforms (jit, vmap, etc).
        """
        self.bvalues = jnp.array(self.bvalues)
        self.gradient_directions = jnp.array(self.gradient_directions)
        
        if self.delta is not None:
            self.delta = jnp.array(self.delta)
        if self.Delta is not None:
            self.Delta = jnp.array(self.Delta)
        if self.echo_time is not None:
            self.echo_time = jnp.array(self.echo_time)
        if self.total_readout_time is not None:
            self.total_readout_time = jnp.array(self.total_readout_time)

        # Initialize B-Tensors
        if self.btensors is None:
            # Construct LTE B-tensors from bvals and bvecs
            # B = bval * outer(g, g)
            # Shapes: bvals (N,), g (N, 3) -> B (N, 3, 3)
            
            # einsum 'n, ni, nj -> nij'
            self.btensors = jnp.einsum('n, ni, nj -> nij', self.bvalues, self.gradient_directions, self.gradient_directions)
        else:
            self.btensors = jnp.array(self.btensors)

    @property
    def qvalues(self):
        """
        Calculates q-values in 1/m.
        q = 1/(2*pi) * sqrt(b / (Delta - delta/3))
        
        Requires delta and Delta to be set.
        """
        if self.delta is None or self.Delta is None:
            # Fallback or error?
            # If b=0, q=0.
            # If delta/Delta missing, cannot compute q from b.
            # Maybe return None or raise Error?
            # For now, raise Error to ensure user provides them for models needing q.
            raise ValueError("JaxAcquisition: delta and Delta must be set to compute qvalues.")
        
        # Calculate diffusion time
        tau = self.Delta - self.delta / 3.0
        
        # Avoid division by zero if tau is zero (unlikely but possible in bad data)
        # q = sqrt(b/tau) / (2pi)
        
        # We need to handle b=0 case safely.
        # If b is 0, q is 0.
        
        q = jnp.sqrt(self.bvalues / tau) / (2 * jnp.pi)
        return q

    def to_device(self, device: Optional[Any] = None) -> 'JaxAcquisition':
        """
        Move all acquisition arrays to the specified JAX device (CPU/GPU/TPU).
        
        Parameters
        ----------
        device : jax.Device, optional
            The target device to move arrays to. If None, JAX chooses the default
            device (usually the first available GPU or TPU).

        Returns
        -------
        JaxAcquisition
            Self, with arrays updated to reside on the target device.
        """
        self.bvalues = jax.device_put(self.bvalues, device)
        self.gradient_directions = jax.device_put(self.gradient_directions, device)
        self.btensors = jax.device_put(self.btensors, device)
        
        if self.delta is not None:
            self.delta = jax.device_put(self.delta, device)
        
        if self.Delta is not None:
            self.Delta = jax.device_put(self.Delta, device)
            
        if self.echo_time is not None:
            self.echo_time = jax.device_put(self.echo_time, device)

        if self.total_readout_time is not None:
            self.total_readout_time = jax.device_put(self.total_readout_time, device)

        return self

    @classmethod
    def from_bids_data(cls, bids_data: Dict[str, Any]) -> 'JaxAcquisition':
        """
        Creates a JaxAcquisition instance from a dictionary of BIDS data.

        Args:
            bids_data: A dictionary containing paths to 'bval_file', 'bvec_file',
                       and optional metadata 'EchoTime', 'TotalReadoutTime'.

        Returns:
            JaxAcquisition: A configured acquisition object.
        """
        bval_file = bids_data.get('bval_file')
        bvec_file = bids_data.get('bvec_file')

        if not bval_file or not bvec_file:
            raise ValueError("bids_data must contain 'bval_file' and 'bvec_file'.")

        # Load text files
        # bvals are typically 1D or 1xN
        bvals = np.loadtxt(bval_file)
        if bvals.ndim == 2:
            bvals = bvals.squeeze()

        # bvecs are typically 3xN in FSL/BIDS
        bvecs = np.loadtxt(bvec_file)
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T

        echo_time = bids_data.get('EchoTime')
        total_readout_time = bids_data.get('TotalReadoutTime')
        
        # Determine delta/Delta if present (unlikely in standard scalar metadata but possible)
        # Using .get for safe access
        delta = bids_data.get('SmallDelta') or bids_data.get('delta')
        Delta = bids_data.get('BigDelta') or bids_data.get('Delta')

        return cls(
            bvalues=bvals,
            gradient_directions=bvecs,
            echo_time=echo_time,
            total_readout_time=total_readout_time,
            delta=delta,
            Delta=Delta
        )

# Register JaxAcquisition as a Pytree node
# This allows JAX to trace through the object and handle its array fields correctly.

def _acquisition_flatten(acq):
    # Children are the dynamic array fields
    children = (
        acq.bvalues,
        acq.gradient_directions,
        acq.delta,
        acq.Delta,
        acq.echo_time,
        acq.total_readout_time,
        acq.btensors
    )
    # Aux data is empty/None as we don't have static metadata that affects the structure
    aux_data = None
    return children, aux_data

def _acquisition_unflatten(aux_data, children):
    # Reconstruct the object
    return JaxAcquisition(
        bvalues=children[0],
        gradient_directions=children[1],
        delta=children[2],
        Delta=children[3],
        echo_time=children[4],
        total_readout_time=children[5],
        btensors=children[6]
    )

jax.tree_util.register_pytree_node(
    JaxAcquisition,
    _acquisition_flatten,
    _acquisition_unflatten
)
