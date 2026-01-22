
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import pickle

# Local imports
from dmipy_jax.core.particle_engine import brownian_step, create_neighbor_list, periodic_box

def run_simulation_batch(key, radius, diffusion_coeff, n_walkers=1000, n_steps=200, dt=0.01):
    """
    Simulates random walkers in a cylinder of given radius.
    Returns the mean signal E(q) for a set of q-values.
    Using Narrow Pulse Approximation: E(q) = < exp(i q . (r_T - r_0)) >
    """
    # Ensure numerical stability for radius
    radius = jnp.maximum(radius, 0.1)
         
    # 1. Initialize Walkers
    # Distributed uniformly inside the cylinder (disk in 2D)
    # Rejection sampling or polar coordinates
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Polar sampling: r = R * sqrt(u), theta = 2pi * v
    u = jax.random.uniform(k1, (n_walkers,))
    theta = jax.random.uniform(k2, (n_walkers,), maxval=2*jnp.pi)
    r = radius * jnp.sqrt(u)
    
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    z = jax.random.uniform(k3, (n_walkers,), minval=0, maxval=10.0) # Free diffusion in Z
    
    positions = jnp.stack([x, y, z], axis=-1)
    start_positions = positions
    
    # 2. Simulation Loop
    def step_fn(pos, k):
        # Brownian Step
        new_pos = brownian_step(k, pos, lambda x, d: x+d, diffusion_coeff, dt)
        
        # Enforce Cylinder Boundary (Reflective) in XY plane
        # calc radial distance
        pos_xy = new_pos[..., :2]
        dist = jnp.linalg.norm(pos_xy, axis=-1)
        
        # Check collision: dist > radius
        mask = dist > radius
        
        # Reflection: 
        # project to boundary and reflect velocity component?
        # Simpler: Mirror projection.
        # r' = r - 2*(r - R_boundary) ? No.
        # Vector r. Normal n = r / |r|.
        # overlap = |r| - R.
        # new_r = r - 2 * overlap * n
        
        # Avoid division by zero
        safe_dist = jnp.where(dist < 1e-6, 1.0, dist)
        normal = pos_xy / safe_dist[..., None]
        
        overlap = dist - radius
        # Only reflect where mask is true
        overlap = jnp.where(mask, overlap, 0.0)
        
        pos_xy_reflected = pos_xy - 2 * overlap[..., None] * normal
        
        # Update XY
        new_pos = new_pos.at[..., :2].set(pos_xy_reflected)
        return new_pos, nil

    nil = jnp.zeros(1)
    keys = jax.random.split(key, n_steps)
    
    # Run scan
    final_positions, _ = jax.lax.scan(step_fn, positions, keys)
    
    # 3. Compute Signal E(q)
    # Displacements
    displacements = final_positions - start_positions
    
    # Define q-space sampling
    # Let's compute for a fixed set of q-values perpendicular to cylinder (measuring restriction)
    # q along X axis.
    # q_values = [0, 1e4, ..., 1e6] m^-1.
    # We want to output E(q) for downstream learning.
    # Actually, we need to return input (radius) and output (Signal vector).
    
    # Vectorized computation for multiple q's
    qs = jnp.linspace(0, 5e5, 50) # 50 q-points
    # q vectors: (50, 3) -> (q, 0, 0)
    q_vecs = jnp.stack([qs, jnp.zeros_like(qs), jnp.zeros_like(qs)], axis=-1)
    
    # exp(i q . dX)
    # q: (Q, 3), dX: (N, 3). Dot: (Q, N)
    phase = jnp.dot(q_vecs, displacements.T)
    signals = jnp.mean(jnp.exp(1j * phase), axis=1)
    
    return jnp.abs(signals) # Magnitude signal

def generate_emulator_data():
    print("=== Generating Emulator Data (JAX-MD) ===")
    
    N_SAMPLES = 200 # Number of geometries (radii)
    # Using vmap to generate data in parallel
    
    radii = jnp.linspace(1e-6, 10e-6, N_SAMPLES) # 1um to 10um
    diffusion_coeff = 2e-9 # 2 um^2/ms
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, N_SAMPLES)
    
    print(f"Simulating {N_SAMPLES} cylinders with radii {radii[0]:.2e} to {radii[-1]:.2e}...")
    
    # Vmap simulation
    # Note: Loop 200 simulation steps inside vmap might be heavy.
    # But JAX handles it.
    
    signals = jax.vmap(run_simulation_batch, in_axes=(0, 0, None))(keys, radii, diffusion_coeff)
    
    print("Simulation Complete.")
    print(f"Signal Shape: {signals.shape} (Samples, Q-points)")
    
    # Save Data
    qs = np.linspace(0, 5e5, 50)
    data = {
        'radii': np.array(radii),
        'q_values': qs, # 1D q's
        'signals': np.array(signals)
    }
    
    with open('emulator_train_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Data saved to emulator_train_data.pkl")

if __name__ == "__main__":
    generate_emulator_data()
