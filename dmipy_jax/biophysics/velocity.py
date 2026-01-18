import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional

def hursh_rushton_velocity(
    axon_diameter: Float[Array, "..."],
    g_ratio: Float[Array, "..."],
    k: float = 5.5
) -> Float[Array, "..."]:
    """
    Calculates conduction velocity using the Hursh/Rushton relation.
    
    V = k * d_fiber
    where d_fiber = d_axon / g_ratio
    
    This implies V = k * (d_axon / g_ratio).
    
    Args:
        axon_diameter: Axon map in micrometers (um).
        g_ratio: g-ratio map (d_axon / d_fiber). Restricted to (0, 1).
        k: Proportionality constant. Default is 5.5 m/s per um (Hursh 1939 value often cited).
           
    Returns:
        Conduction velocity in m/s.
    """
    # Avoid division by zero if g_ratio is 0.
    # g_ratio usually ~0.6-0.8.
    
    fiber_diameter = axon_diameter / (g_ratio + 1e-12)
    velocity = k * fiber_diameter
    return velocity

def calculate_latency_matrix(
    tract_lengths: Float[Array, "n_regions n_regions"],
    mean_velocities: Float[Array, "n_regions n_regions"]
) -> Float[Array, "n_regions n_regions"]:
    """
    Computes the latency matrix given tract lengths and mean velocities per connection.
    
    Latency t = L / V.
    
    Args:
        tract_lengths: Matrix of connection lengths in millimeters (mm).
        mean_velocities: Matrix of mean conduction velocities in meters per second (m/s).
        
    Returns:
        Latency matrix in milliseconds (ms).
    """
    # 1 m/s = 1000 mm / 1000 ms = 1 mm/ms.
    # So if L is in mm and V is in m/s, t = L / V is directly in ms.
    
    # helper for safe division
    # If velocity is 0 or very small, latency implies infinity/disconnected.
    # We'll set 0 velocity -> 0 latency (or infinity? standard is usually 0 for unconnected in adjacency matrices, 
    # but for latency specifically, infinite makes more physical sense. 
    # However, in "latency matrices" for source modeling, 0 often implies no connection/delay=0 or handled by mask.
    # Let's return inf for now where velocity is 0, unless lengths is also 0.
    
    # Actually, simpler: just divide.
    
    safe_v = jnp.where(mean_velocities > 1e-6, mean_velocities, jnp.inf)
    
    latencies = tract_lengths / safe_v
    
    # If length is 0, latency is 0 (self-connection or no connection)
    latencies = jnp.where(tract_lengths == 0, 0.0, latencies)
    
    # If result is inf (disconnected), keep it inf or let user handle?
    # Usually sparse matrices or masked.
    
    return latencies
