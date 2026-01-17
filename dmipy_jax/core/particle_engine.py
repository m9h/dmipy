import jax
import jax.numpy as jnp
from jax_md import space, partition, simulate
from typing import Tuple, Callable, Optional, Union

# --- Box Definitions ---

def check_gpu_visibility():
    """Checks if JAX can see the GPU and prints the status.
    
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    devices = jax.devices()
    gpu_available = any(device.platform == 'gpu' for device in devices)
    if gpu_available:
        print(f"✅ GPU found: {[d for d in devices if d.platform == 'gpu']}")
    else:
        print("⚠️ No GPU found. Running on CPU.")
    return gpu_available

def check_gpu_visibility():
    """Checks if JAX can see the GPU and prints the status.
    
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    devices = jax.devices()
    gpu_available = any(device.platform == 'gpu' for device in devices)
    if gpu_available:
        print(f"✅ GPU found: {[d for d in devices if d.platform == 'gpu']}")
    else:
        print("⚠️ No GPU found. Running on CPU.")
    return gpu_available

def periodic_box(box_size: float) -> Tuple[Callable, Callable]:
    """Defines a periodic box for tissue simulations.
    
    Args:
        box_size: The side length of the cubic box.
        
    Returns:
        (displacement_fn, shift_fn) pair compatible with jax-md.
    """
    return space.periodic(box_size)

def non_periodic_box() -> Tuple[Callable, Callable]:
    """Defines a non-periodic (infinite) box for whole-brain tracking.
    
    Returns:
        (displacement_fn, shift_fn) pair compatible with jax-md.
    """
    return space.free()

# --- Walker (Brownian Motion) ---

def brownian_step(
    key: jax.Array,
    positions: jax.Array,
    shift_fn: Callable,
    diffusion_coeff: Union[float, jax.Array],
    dt: float,
    kT: float = 1.0 # Included for jax-md signature compatibility, though often 1.0 in normalized units
) -> jax.Array:
    """Computes one step of Brownian motion with potentially anisotropic diffusion.

    Adapted from jax_md.simulate.brownian but supports Tensor diffusion coefficients.
    
    Args:
        key: PRNGKey for noise generation.
        positions: (N, dim) array of particle positions.
        shift_fn: Function to apply boundary conditions.
        diffusion_coeff: 
            - Scalar: Isotropic, homogenous diffusion.
            - (3,) Vector: Anisotropic diagonal diffusion (axis-aligned).
            - (N, 3, 3) Tensor: Fully anisotropic diffusion per particle. 
              (Note: Spatially varying maps would need a lookup mechanism, 
               here we assume D is provided per-particle or globally).
        dt: Time step.
        kT: Temperature (default 1.0).

    Returns:
        New positions (N, dim).
    """
    
    # Generate Gaussian noise
    noise = jax.random.normal(key, positions.shape, dtype=positions.dtype)
    
    # Calculate displacement: dX = sqrt(2 * D * dt) * noise
    
    # Case 1: Scalar or Vector (Diagonal) Diffusion
    if jnp.ndim(diffusion_coeff) == 0 or (jnp.ndim(diffusion_coeff) == 1 and diffusion_coeff.shape[0] == positions.shape[1]):
        scale = jnp.sqrt(2 * diffusion_coeff * dt)
        dr = noise * scale
        
    # Case 2: Full Tensor Diffusion (N, 3, 3) or (3, 3)
    elif jnp.ndim(diffusion_coeff) >= 2:
        # We need L such that L @ L.T = D. 
        # For Brownian motion, variance is 2*D*dt.
        # So we need transform matrix M = sqrt(2*dt) * L
        # Then dX = M @ noise
        
        # Ensure diffusion_coeff is at least (1, 3, 3) if global, or (N, 3, 3)
        D = diffusion_coeff
        if D.ndim == 2:
            D = D[None, :, :] # Broadcast global tensor
            
        # Cholesky decomposition: D = L L^T
        # We need to handle potential numerical issues if D is not strictly positive definite
        # Adding a small epsilon or using eigh might be safer, but Cholesky is fastest.
        try:
             L = jnp.linalg.cholesky(D)
        except:
             # Fallback to Eigendecomposition if Cholesky fails (e.g. semi-definite)
             # D = V S V^T -> L = V sqrt(S)
            eigs, vecs = jnp.linalg.eigh(D)
            # Clip negative eigenvalues needed for numerical stability
            eigs = jnp.maximum(eigs, 0.) 
            L = vecs * jnp.sqrt(eigs)[..., None, :]

        scale = jnp.sqrt(2 * dt)
        
        # dX_i = sum_j L_ij * noise_j
        # Batched matrix-vector multiplication
        # noise is (N, 3), L is (N, 3, 3) or (1, 3, 3)
        # We want result (N, 3)
        
        # If D is (N, 3, 3) and noise is (N, 3)
        # einsum 'nij,nj->ni'
        dr = scale * jnp.einsum('nij,nj->ni', L, noise)

    else:
        raise ValueError(f"Unsupported shape for diffusion_coeff: {diffusion_coeff.shape}")

    return shift_fn(positions, dr)


# --- Neighbor List ---

def create_neighbor_list(
    displacement_fn: Callable,
    box_size: Union[float, jax.Array],
    r_cutoff: float,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False
) -> partition.NeighborList:
    """Creates a basic neighbor list for collision/barrier detection.

    Args:
        displacement_fn: Function to compute displacement (metric).
        box_size: The size of the simulation box.
        r_cutoff: Cutoff distance for neighbors.
        capacity_multiplier: Multiplier for initial capacity estimation.
        disable_cell_list: If True, uses O(N^2) global search (good for small N).
        
    Returns:
        An `init_fn, apply_fn` tuple (or NeighborList object in newer jax-md).
        You typically call `.allocate(positions)` on this.
    """
    
    return partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        format=partition.Dense # Dense is usually easier to work with for fixed max neighbors
    )
