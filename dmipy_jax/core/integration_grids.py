
import jax
import jax.numpy as jnp

__all__ = ['get_spherical_fibonacci_grid']

def get_spherical_fibonacci_grid(n_points=100):
    """
    Generates a Spherical Fibonacci Lattice of N points.
    
    This is an approximate "equal area" grid on the sphere, suitable for 
    numerical integration where strict quadrature rules (like Lebedev) 
    are not strictly required or are too heavy to implement inline.
    
    Args:
        n_points (int): Number of points on the sphere.
        
    Returns:
        points (jnp.ndarray): (N, 3) array of Cartesian coordinates on unit sphere.
        weights (jnp.ndarray): (N,) array of integration weights. For Fibonacci,
                               these are approximately uniform (4*pi / N).
    """
    # Golden ratio
    phi = (1 + jnp.sqrt(5)) / 2

    i = jnp.arange(n_points)
    
    # Z coordinate ranges from 1 to -1 (or close to it)
    # Mapping i to [-1, 1]
    # Standard choice: z = 1 - (2*i + 1)/n
    z = 1 - (2 * i + 1) / n_points
    
    # Radius at z
    radius = jnp.sqrt(1 - z * z)
    
    # Azimuthal angle
    theta = 2 * jnp.pi * i / phi
    
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    
    points = jnp.stack([x, y, z], axis=1)
    
    # For Fibonacci lattice, area elements are approximately equal
    # Total area of sphere is 4*pi
    weights = jnp.ones(n_points) * (4 * jnp.pi / n_points)
    
    return points, weights
