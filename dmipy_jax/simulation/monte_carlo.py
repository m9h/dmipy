"""Ground-truth Monte Carlo random-walk simulator with SDF-based geometry.

Provides a particle-tracking Monte Carlo engine for simulating restricted
and hindered diffusion inside arbitrary geometries defined by signed distance
functions (SDFs).  This module serves as the reference ("gold standard")
simulator against which the analytical models and FEM/SDE simulators are
validated.

Key functions:

* :func:`reflect` -- elastic collision reflection for particles that step
  outside the SDF boundary.
* :func:`step_particles` -- single time-step function compatible with
  ``jax.lax.scan``; advances particle positions, applies boundary
  reflections, and accumulates spin phase under the applied gradient.
* :func:`run_monte_carlo` -- top-level entry point that runs N particles
  for a full PGSE sequence and returns the ensemble-averaged signal.

The simulator operates in SI units and uses the proton gyromagnetic ratio
(gamma = 2.6752e8 rad/s/T) for phase accumulation.

See Also:
    :mod:`dmipy_jax.simulation.sphere_sdf` -- sphere SDF primitives used
    as geometry inputs.
    :mod:`dmipy_jax.simulation.differentiable_walker` -- differentiable
    alternative for gradient-based optimisation.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

def reflect(position, old_position, sdf_func, sdf_grad=None):
    """
    Elastic collision reflection for a particle stepping outside the SDF.

    Args:
        position: The proposed new position (inside the wall, SDF > 0).
        old_position: The previous position (valid, SDF <= 0).
        sdf_func: The Signed Distance Function.
        sdf_grad: Pre-computed gradient function of sdf_func (optional).

    Returns:
        reflected_position: The position reflected back into the valid region.
    """
    # 1. Compute surface normal at the collision point.
    # We approximate the collision point as the 'position' for gradient calculation.
    # In a strict sense, we should find the root, but for small steps, this is sufficient.
    if sdf_grad is None:
        sdf_grad = grad(sdf_func)
    normal = sdf_grad(position)
    
    # Normalize the normal vector
    normal = normal / (jnp.linalg.norm(normal) + 1e-12)
    
    # 2. Reflect the vector.
    # The penetration depth is approximately sdf(position).
    # We need to move back by 2 * depth * normal to reflect elastically.
    dist = sdf_func(position)
    reflection = position - 2 * dist * normal
    
    return reflection

def step_particles(state, input_data, sdf_func, D, dt, confinement='inside', gamma=2.6751525e8):
    """
    Single time-step function for jax.lax.scan.
    
    Args:
        state: tuple (positions, accumulated_phase, key)
        input_data: tuple (gradient_vector_at_t, )
        sdf_func: function(position) -> distance
        D: Diffusion coefficient
        dt: Time step duration
        confinement: 'inside' (restricted) or 'outside' (hindered).
        gamma: Gyromagnetic ratio (default: Proton)
        
    Returns:
        new_state: Updated state
        output: None (or debug info)
    """
    positions, phase, key = state
    g_t = input_data # Gradient vector at current time step (3,)

    key, subkey = jax.random.split(key)

    # Pre-compute gradient function once for this geometry
    sdf_grad = grad(sdf_func)

    # 1. Brownian Step
    noise = jax.random.normal(subkey, shape=positions.shape)
    step = jnp.sqrt(2 * D * dt) * noise
    proposed_positions = positions + step

    # 2. Check collisions and Reflect
    def check_and_reflect(pos, old_pos):
        dist = sdf_func(pos)

        # Logic:
        # inside: valid if <= 0. Invalid if > 0.
        # outside: valid if > 0. Invalid if <= 0.
        if confinement == 'inside':
             is_invalid = dist > 0
        else:
             is_invalid = dist <= 0

        return jax.lax.cond(
            is_invalid,
            lambda p, op: reflect(p, op, sdf_func, sdf_grad),  # True branch
            lambda p, op: p,                                     # False branch
            pos, old_pos
        )

    new_positions = vmap(check_and_reflect)(proposed_positions, positions)
    
    # 3. Accumulate Phase
    phase_increment = gamma * jnp.dot(new_positions, g_t) * dt
    new_phase = phase + phase_increment
    
    return (new_positions, new_phase, key), None


def simulate_ground_truth(geometry_sdf, initialization_func, gamma=2.6751525e8, confinement='inside'):
    """
    Creates a simulation function for a specific geometry.
    
    Args:
        geometry_sdf: Function taking (x, y, z) or (3,) array returning signed distance.
                      Convention: <= 0 is INSIDE (fluid), > 0 is OUTSIDE (wall).
        initialization_func: Function taking (key, n_particles) returning (n, 3) initial positions.
        gamma: Gyromagnetic ratio (Hz/T).
        confinement: 'inside' or 'outside'.
        
    Returns:
        simulate: Function(gradient_waveform, D, dt, N_particles, key) -> Signal
    """
    
    # JIT-compile the scanner for this geometry
    @partial(jit, static_argnums=(3,))
    def simulate(gradient_waveform, D, dt, N_particles, key):
        """
        Run the simulation.
        """
        # 1. Initialize Particles
        key, subkey = jax.random.split(key)
        initial_positions = initialization_func(subkey, N_particles)
        initial_phase = jnp.zeros(N_particles)
        
        init_state = (initial_positions, initial_phase, key)
        
        # 2. Run Scan Loop
        step_fn = partial(step_particles, sdf_func=geometry_sdf, D=D, dt=dt, confinement=confinement, gamma=gamma)
        
        final_state, _ = jax.lax.scan(step_fn, init_state, gradient_waveform)
        
        # 3. Compute Signal
        final_phases = final_state[1] # (N_particles,)
        ensemble_signal = jnp.mean(jnp.exp(1j * final_phases))
        
        return jnp.abs(ensemble_signal)

    return simulate


def step_particles_trajectory(state, input_data, sdf_func, D, dt, confinement='inside', gamma=2.6751525e8):
    """
    Same as step_particles but returns positions for trajectory recording.
    """
    positions, phase, key = state
    g_t = input_data # Gradient vector at current time step (3,)
    
    key, subkey = jax.random.split(key)
    
    # 1. Brownian Step
    noise = jax.random.normal(subkey, shape=positions.shape)
    step = jnp.sqrt(2 * D * dt) * noise
    proposed_positions = positions + step
    
    # 2. Check collisions and Reflect
    def check_and_reflect(pos, old_pos):
        dist = sdf_func(pos)
        
        if confinement == 'inside':
             is_invalid = dist > 0
        else:
             is_invalid = dist <= 0
             
        return jax.lax.cond(
            is_invalid,
            lambda p, op: reflect(p, op, sdf_func),
            lambda p, op: p,
            pos, old_pos
        )

    new_positions = vmap(check_and_reflect)(proposed_positions, positions)
    
    # 3. Accumulate Phase
    phase_increment = gamma * jnp.dot(new_positions, g_t) * dt
    new_phase = phase + phase_increment
    
    # Return new_positions as the scanned output
    return (new_positions, new_phase, key), new_positions

def simulate_trajectories(geometry_sdf, initialization_func, gamma=2.6751525e8, confinement='inside'):
    """
    Creates a simulation function that returns TRAJECTORIES.
    """
    
    @partial(jit, static_argnums=(2, 3))
    def simulate(D, dt, N_particles, N_steps, key):
        """
        Run simulation and return trajectories.
        """
        # 1. Initialize
        key, subkey = jax.random.split(key)
        initial_positions = initialization_func(subkey, N_particles)
        initial_phase = jnp.zeros(N_particles)
        
        init_state = (initial_positions, initial_phase, key)
        
        # Dummy gradient (zeros) as we just want diffusion trajectories
        dummy_gradient = jnp.zeros((N_steps, 3))
        
        # 2. Run Scan
        step_fn = partial(step_particles_trajectory, sdf_func=geometry_sdf, D=D, dt=dt, confinement=confinement, gamma=gamma)
        
        final_state, trajectories = jax.lax.scan(step_fn, init_state, dummy_gradient)
        
        return initial_positions, trajectories
        
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
