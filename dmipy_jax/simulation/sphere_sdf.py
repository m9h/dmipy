"""Signed distance functions (SDFs) for spherical geometry primitives.

Provides SDF building blocks for defining restricted-diffusion compartments
in the Monte Carlo and SDE simulators.  SDFs return negative values inside
the boundary and positive values outside, with zero on the surface.

Key components:

* :func:`sphere_sdf` -- SDF for a single sphere centred at ``center`` with
  the given ``radius``.
* :class:`MultiSphereSDF` -- efficient union SDF for N spheres, computing
  ``min_i(SDF_i(p))`` in a single vectorised pass.  Useful for packed-sphere
  geometries (e.g. SANDI soma models).

Both are compatible with ``jax.grad`` for computing surface normals, which
the Monte Carlo reflector (:func:`~dmipy_jax.simulation.monte_carlo.reflect`)
uses for elastic boundary collisions.
"""

import jax
import jax.numpy as jnp
from functools import partial

def sphere_sdf(position, center, radius):
    """
    Signed Distance Function for a single sphere.
    dist = ||p - c|| - r
    <= 0 inside, > 0 outside.
    """
    return jnp.linalg.norm(position - center) - radius

class MultiSphereSDF:
    """
    Efficient SDF for a union of many spheres.
    SDF_union(p) = min_i (SDF_i(p))
    """
    def __init__(self, centers, radii):
        """
        Args:
            centers: (N, 3) array of sphere centers.
            radii: (N,) array of radii.
        """
        self.centers = jnp.asarray(centers)
        self.radii = jnp.asarray(radii)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, position):
        """
        Computes the SDF value at a single position.
        
        Args:
            position: (3,) array.
            
        Returns:
            Scalar distance.
        """
        # Distance to all sphere centers
        # d_centers = ||p - c_i||
        diff = position - self.centers # (N, 3)
        d_centers = jnp.linalg.norm(diff, axis=1) # (N,)
        
        # SDF_i = d_centers_i - r_i
        sdf_values = d_centers - self.radii # (N,)
        
        # Union = min(SDF_i)
        return jnp.min(sdf_values)

    def get_sdf_func(self):
        """
        Returns a pure function `sdf(position)` that closes over the data.
        Useful for passing to simulators that expect a function.
        """
        centers = self.centers
        radii = self.radii
        
        def sdf(position):
            diff = position - centers
            d_centers = jnp.linalg.norm(diff, axis=1)
            return jnp.min(d_centers - radii)
        
        return sdf
