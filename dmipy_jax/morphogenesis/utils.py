"""
Utility functions for the Morphogenesis pipeline.
Converts dmipy_jax microstructural parameters to mechanical properties.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

def map_microstructure_to_stiffness(
    f_ic: jax.Array, 
    f_sphere: jax.Array, 
    base_stiffness: float = 1.0
) -> jax.Array:
    """
    Empirical mapping from volume fractions to stiffness (shear modulus proxy).
    
    Higher axonal density (f_ic) increases white matter stiffness.
    Higher soma density (f_sphere) increases cortical stiffness.
    """
    # Simple linear mapping as a starting point
    # White matter: mu ~ mu0 * (1 + beta * f_ic)
    # Gray matter: mu ~ mu0 * (1 + gamma * f_sphere)
    
    # We use a combined approach for a voxel-based map
    # Stiffness increases with any cellular density
    return base_stiffness * (1.0 + 2.0 * f_ic + 3.0 * f_sphere)

def get_growth_tensor(
    growth_ratio: jax.Array, 
    orientation: jax.Array, 
    is_tangential: bool = True
) -> jax.Array:
    """
    Constructs a growth deformation gradient tensor Fg.
    
    Args:
        growth_ratio: Scalar or field of growth magnitudes.
        orientation: Fiber direction a0 (used to define the radial/tangential plane).
        is_tangential: If True, growth occurs perpendicular to the fiber (cortical expansion).
                       If False, growth occurs along the fiber (axonal elongation).
                       
    Returns:
        Fg: (3, 3, ...) growth tensor.
    """
    # For tangential growth (cortical expansion):
    # Fg = growth_ratio * (I - a0 x a0) + 1.0 * (a0 x a0)
    
    I = jnp.eye(3)
    # Outer product a0 x a0
    # a0 shape: (3, ...)
    A = jnp.einsum('i...,j...->ij...', orientation, orientation)
    
    if is_tangential:
        # Expand in the plane perpendicular to a0
        # Fg = g * (I - A) + 1.0 * A
        Fg = growth_ratio * (I[..., None, None, None] - A) + 1.0 * A
    else:
        # Elongate along a0
        # Fg = 1.0 * (I - A) + g * A
        Fg = 1.0 * (I[..., None, None, None] - A) + growth_ratio * A
        
    return Fg
