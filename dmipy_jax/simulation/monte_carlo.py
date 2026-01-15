
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

def reflect(position, old_position, sdf_func):
    """
    Elastic collision reflection for a particle stepping outside the SDF.
    
    Args:
        position: The proposed new position (inside the wall, SDF > 0).
        old_position: The previous position (valid, SDF <= 0).
        sdf_func: The Signed Distance Function.
        
    Returns:
        reflected_position: The position reflected back into the valid region.
    """
    # 1. Compute surface normal at the collision point. 
    # We approximate the collision point as the 'position' for gradient calculation.
    # In a strict sense, we should find the root, but for small steps, this is sufficient.
    # We use jax.grad to get the gradient of the SDF.
    normal = grad(sdf_func)(position)
    
    # Normalize the normal vector
    normal = normal / (jnp.linalg.norm(normal) + 1e-12)
    
    # 2. Reflect the vector.
    # The penetration depth is approximately sdf(position).
    # We need to move back by 2 * depth * normal to reflect elastically.
    dist = sdf_func(position)
    reflection = position - 2 * dist * normal
    
    return reflection

def step_particles(state, input_data, sdf_func, D, dt, gamma=2.6751525e8):
    """
    Single time-step function for jax.lax.scan.
    
    Args:
        state: tuple (positions, accumulated_phase, key)
        input_data: tuple (gradient_vector_at_t, )
        sdf_func: function(position) -> distance
        D: Diffusion coefficient
        dt: Time step duration
        gamma: Gyromagnetic ratio (default: Proton)
        
    Returns:
        new_state: Updated state
        output: None (or debug info)
    """
    positions, phase, key = state
    g_t = input_data # Gradient vector at current time step (3,)
    
    # Split key for random generation
    key, subkey = jax.random.split(key)
    
    # 1. Brownian Step: dX = sqrt(2*D*dt) * N(0, 1)
    noise = jax.random.normal(subkey, shape=positions.shape)
    step = jnp.sqrt(2 * D * dt) * noise
    proposed_positions = positions + step
    
    # 2. Check collisions and Reflect
    # We vectorise the SDF check and reflection logic over all particles
    
    # Define a single particle check/reflect function
    def check_and_reflect(pos, old_pos):
        dist = sdf_func(pos)
        # If dist > 0, we are outside (assuming SDF > 0 is outside/wall)
        # We use jax.lax.cond to conditionally reflect
        
        # NOTE: This assumes sdf_func defines "inside" as <= 0.
        is_outside = dist > 0
        
        return jax.lax.cond(
            is_outside,
            lambda p, op: reflect(p, op, sdf_func), # True branch (reflect)
            lambda p, op: p,                        # False branch (keep)
            pos, old_pos
        )

    new_positions = vmap(check_and_reflect)(proposed_positions, positions)
    
    # 3. Accumulate Phase
    # d_phase = gamma * (g_t . x) * dt
    # g_t is (3,), x is (N, 3). dot product over last axis.
    phase_increment = gamma * jnp.dot(new_positions, g_t) * dt
    new_phase = phase + phase_increment
    
    return (new_positions, new_phase, key), None


def simulate_ground_truth(geometry_sdf, initialization_func, gamma=2.6751525e8):
    """
    Creates a simulation function for a specific geometry.
    
    Args:
        geometry_sdf: Function taking (x, y, z) or (3,) array returning signed distance.
                      Convention: <= 0 is INSIDE (fluid), > 0 is OUTSIDE (wall).
        initialization_func: Function taking (key, n_particles) returning (n, 3) initial positions.
        gamma: Gyromagnetic ratio (Hz/T).
        
    Returns:
        simulate: Function(gradient_waveform, D, dt, N_particles, key) -> Signal
    """
    
    # JIT-compile the scanner for this geometry
    # N_particles (arg 3) must be static because it determines array shapes.
    @partial(jit, static_argnums=(3,))
    def simulate(gradient_waveform, D, dt, N_particles, key):
        """
        Run the simulation.
        
        Args:
            gradient_waveform: (N_steps, 3) array of gradients [T/m].
            D: Diffusion coefficient [m^2/s].
            dt: Time step [s].
            N_particles: Number of walkers.
            key: JAX PRNGKey.
            
        Returns:
            Signal: Complex signal attenuation (scalar).
        """
        # 1. Initialize Particles
        key, subkey = jax.random.split(key)
        initial_positions = initialization_func(subkey, N_particles)
        initial_phase = jnp.zeros(N_particles)
        
        init_state = (initial_positions, initial_phase, key)
        
        # 2. Run Scan Loop
        # Partial application of the step function with fixed params
        step_fn = partial(step_particles, sdf_func=geometry_sdf, D=D, dt=dt, gamma=gamma)
        
        final_state, _ = jax.lax.scan(step_fn, init_state, gradient_waveform)
        
        # 3. Compute Signal
        # Signal = | sum(exp(i * phase)) | / N
        final_phases = final_state[1] # (N_particles,)
        
        # This is the complex signal
        ensemble_signal = jnp.mean(jnp.exp(1j * final_phases))
        
        return jnp.abs(ensemble_signal)

    return simulate


# --- Helper SDFs ---

def cylinder_sdf(position, radius, center=jnp.array([0., 0.])):
    """
    SDF for an infinite cylinder along Z-axis.
    Convention: <= 0 inside, > 0 outside.
    """
    # Assuming position is (3,)
    xy = position[:2] - center
    dist = jnp.linalg.norm(xy) - radius
    return dist

def sphere_sdf(position, radius, center=jnp.array([0., 0., 0.])):
    """
    SDF for a sphere.
    """
    dist = jnp.linalg.norm(position - center) - radius
    return dist

def parallel_plates_sdf(position, width, normal_axis=0):
    """
    SDF for parallel plates (1D restriction).
    width: distance between plates. Centers at 0.
    """
    # Restricted in one dimension
    x = position[normal_axis]
    dist = jnp.abs(x) - (width / 2.0)
    return dist
