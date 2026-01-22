
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from typing import Tuple, Union

def coord_to_voxel(coords: jnp.ndarray, affine: jnp.ndarray) -> jnp.ndarray:
    """
    Maps physical coordinates (x, y, z) to voxel indices (i, j, k) using the affine matrix.
    
    Args:
        coords: Physical coordinates of shape (..., 3) or (3,).
        affine: 4x4 affine transformation matrix mapping voxel indices to physical coordinates.
        
    Returns:
        Voxel indices of the same shape as input coords (except last dim is 3).
        Note: These are continuous indices (floats), suitable for interpolation.
    """
    # Inverse affine: Physical -> Voxel
    inv_affine = jnp.linalg.inv(affine)
    
    # Check if input is a single point or batch
    is_single = coords.ndim == 1
    if is_single:
        coords = coords[None, :] # (1, 3)
        
    # Append 1 for homogeneous coordinates: (N, 4)
    ones = jnp.ones((coords.shape[0], 1))
    coords_homo = jnp.hstack([coords, ones])
    
    # Transform: (InvAffine @ Coords.T).T -> (Coords @ InvAffine.T)
    # voxel_homo = coords_homo @ inv_affine.T
    
    # More robust: (4, 4) @ (4, N) -> (4, N)
    voxel_homo = inv_affine @ coords_homo.T
    
    # Extract first 3 rows: (3, N) -> Transpose to (N, 3)
    voxel_indices = voxel_homo[:3, :].T
    
    if is_single:
        return voxel_indices[0]
    return voxel_indices

def interpolate_field(field: jnp.ndarray, coords: jnp.ndarray, affine: jnp.ndarray, order: int = 1) -> jnp.ndarray:
    """
    Interpolates a 3D field (scalar or vector) at the given physical coordinates.
    
    Args:
        field: The 3D data array. 
               If scalar field: shape (H, W, D).
               If vector field: shape (C, H, W, D) where C is vector dimension (usually 3).
        coords: Physical coordinates at which to sample, shape (N, 3).
        affine: 4x4 affine matrix.
        order: Interpolation order (0=nearest, 1=linear). Default 1.
        
    Returns:
        Sampled values of shape (N,) for scalar field or (N, C) for vector field.
    """
    # 1. Convert physical coords to voxel indices
    voxel_indices = coord_to_voxel(coords, affine) # (N, 3)
    
    # 2. Prepare coordinates for map_coordinates
    # map_coordinates expects coords of shape (ndim, N_points), i.e., (3, N)
    map_coords = voxel_indices.T 
    
    # 3. Check if vector field (4D) or scalar field (3D)
    if field.ndim == 3:
        # Scalar field: (H, W, D)
        # map_coordinates Input: input (H, W, D), coordinates (3, N)
        samples = map_coordinates(field, map_coords, order=order, mode='nearest')
        return samples # (N,)
        
    elif field.ndim == 4:
        # Vector field: (C, H, W, D). Usually C=3.
        # We need to map each channel separately or vmap over channels.
        
        # Helper to map a single 3D slice
        def map_slice(slice_3d):
            return map_coordinates(slice_3d, map_coords, order=order, mode='nearest')
            
        # vmap over the first dimension (channels)
        # field shape: (C, H, W, D) -> map_slice takes (H, W, D)
        # Output will be (C, N)
        samples = jax.vmap(map_slice)(field)
        
        return samples.T # Returns (N, C)
        
    else:
        raise ValueError(f"Field must be 3D or 4D, got ndim={field.ndim}")
