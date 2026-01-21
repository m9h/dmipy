
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Tuple, List, Optional

class DifferentiableWalker(eqx.Module):
    """
    Differentiable Random Walker for Restricted Geometries.
    
    Uses domain reparameterization to ensure differentiability w.r.t geometry parameters (radius).
    Simulates on a Unit Domain (Sphere/Cylinder of radius 1) with scaled diffusivity,
    then scales the trajectory back to physical dimensions.
    """
    diffusivity: Float[Array, ""]
    radius: Float[Array, ""]
    geometry_type: str = eqx.field(static=True)

    def __init__(self, diffusivity: float, radius: float, geometry_type: str = 'sphere'):
        self.diffusivity = jnp.asarray(diffusivity)
        self.radius = jnp.asarray(radius)
        if geometry_type not in ['sphere', 'cylinder']:
            raise ValueError("geometry_type must be 'sphere' or 'cylinder'")
        self.geometry_type = geometry_type

    def __call__(self, t_span: Tuple[float, float], y0: Float[Array, "dim"], dt: float, key: PRNGKeyArray) -> Float[Array, "steps dim"]:
        """
        Simulate a single particle trajectory.
        
        Args:
            t_span: (t_start, t_end)
            y0: Initial physical position (N_dim,)
            dt: Time step size
            key: PRNGKey
            
        Returns:
            Physical trajectory of shape (num_steps, dim)
        """
        # 1. Reparameterization: Scale to Unit Domain
        # D_unit = D_phys / R^2
        D_unit = self.diffusivity / (self.radius ** 2)
        
        # Scale initial position to unit domain: y0_unit = y0_phys / R
        y0_unit = y0 / self.radius
        
        # 2. Simulation Loop setup
        t0, t1 = t_span
        num_steps = int((t1 - t0) / dt)
        
        # Scan function for the loop
        def step_fn(carry, _):
            pos, current_key = carry
            step_key, next_key = jax.random.split(current_key)
            
            # --- SDE Step (Euler-Heun / Euler-Maruyama for constant noise) ---
            # dX = 0 + sigma * dW
            # sigma = sqrt(2 * D_unit)
            # dw ~ N(0, dt) = N(0, 1) * sqrt(dt)
            dim = pos.shape[0]
            sigma = jnp.sqrt(2 * D_unit)
            noise = jax.random.normal(step_key, shape=(dim,))
            diffusion_step = sigma * noise * jnp.sqrt(dt)
            
            # Proposed new position (Euler steps)
            pos_new = pos + diffusion_step
            
            # --- Reflection Boundaries (on Unit Domain) ---
            if self.geometry_type == 'sphere':
                # Sphere: |x| <= 1
                dist = jnp.linalg.norm(pos_new)
                # Reflection if dist > 1
                # x_reflected = (2 / dist - 1) * x (approximation for small steps?)
                # Correct projection back reflection:
                # If outside, project back: x_new = x_new / dist  (to surface)
                # Then flip the excess: (dist - 1). 
                # Standard formula: x_refl = x_boundary - (x_outside - x_boundary) = 2*x_boundary - x_outside
                # x_boundary = x_outside / dist
                # x_refl = 2 * (x_outside / dist) - x_outside = x_outside * (2/dist - 1)
                
                # Soft logic using jax.numpy.where to allow gradients?
                # Actually, standard `where` is differentiable almost everywhere.
                is_outside = dist > 1.0
                pos_reflected = pos_new * (2.0 / dist - 1.0)
                pos_final = jnp.where(is_outside, pos_reflected, pos_new)
                
            elif self.geometry_type == 'cylinder':
                # Cylinder: x^2 + y^2 <= 1 (assuming z is axis 2)
                # Works for generic dim, assuming last dim is axis? 
                # Standard: Cylinder aligned with Z-axis.
                # Radial part: pos[:2]
                pos_perp = pos_new[:2]
                dist_perp = jnp.linalg.norm(pos_perp)
                
                is_outside = dist_perp > 1.0
                
                # Reflect perpendicular component
                pos_perp_refl = pos_perp * (2.0 / dist_perp - 1.0)
                pos_perp_final = jnp.where(is_outside, pos_perp_refl, pos_perp)
                
                # Combine with parallel component (unchanged)
                pos_final = jnp.concatenate([pos_perp_final, pos_new[2:]])
            else:
                 pos_final = pos_new # Should not happen
                 
            return (pos_final, next_key), pos_final

        # Run the scan
        _, trajectory_unit = jax.lax.scan(step_fn, (y0_unit, key), None, length=num_steps)
        
        # Prepend initial position (optional, but good for completeness)
        # For simplicity return just the steps
        
        # 3. Rescale Output: X_phys = X_unit * R
        trajectory_phys = trajectory_unit * self.radius
        
        return trajectory_phys

def solve_differentiable_walker_batch(
    walker: DifferentiableWalker,
    t_span: Tuple[float, float],
    y0_batch: Float[Array, "N dim"],
    dt: float,
    key: PRNGKeyArray
):
    """
    Batched simulation.
    """
    batch_size = y0_batch.shape[0]
    keys = jax.random.split(key, batch_size)
    
    # vmap over batch
    return jax.vmap(lambda y0, k: walker(t_span, y0, dt, k))(y0_batch, keys)


if __name__ == "__main__":
    print("Running Verification for DifferentiableWalker...")
    
    # 1. Setup
    radius_init = 5.0 # microns
    D_phys = 2.0 # um^2/ms
    T_max = 10.0 # ms
    dt = 0.1
    dim = 3
    N_particles = 1000
    
    # Target function to differentiate: Final MSD
    def compute_msd_at_radius(r_val):
        # Instantiate walker with tracked radius
        walker = DifferentiableWalker(
            diffusivity=D_phys,
            radius=r_val,
            geometry_type='sphere'
        )
        
        # Fixed key for deterministic differentiability
        key = jax.random.PRNGKey(42)
        y0_batch = jnp.zeros((N_particles, dim)) 
        
        # Run simulation
        trajs = solve_differentiable_walker_batch(walker, (0, T_max), y0_batch, dt, key)
        
        # Calculate MSD at final time
        final_pos = trajs[:, -1, :]
        disp = final_pos - y0_batch
        sq_disp = jnp.sum(disp**2, axis=1)
        mean_sq_disp = jnp.mean(sq_disp)
        
        return mean_sq_disp

    # 2. Compute Value and Gradient
    print(f"Computing gradients at R = {radius_init}...")
    loss_and_grad = jax.value_and_grad(compute_msd_at_radius)
    loss_val, grad_val = loss_and_grad(radius_init)
    
    print(f"MSD(R={radius_init}) = {loss_val:.4f}")
    print(f"d(MSD)/dR = {grad_val:.4f}")
    
    # 3. Verification Claims
    # As R increases, confinement is less strict -> particles can travel further -> MSD increases.
    # Therefore, d(MSD)/dR should be Positive.
    
    if grad_val > 0.0:
        print("SUCCESS: Gradient is positive as expected (larger radius = larger displacement).")
        print("Differentiation through the random walk (via reparameterization) is working!")
    else:
        print("FAILURE: Gradient is non-positive. Something is wrong.")
        
    # 4. Check Finite Difference
    delta = 0.01
    loss_plus = compute_msd_at_radius(radius_init + delta)
    loss_minus = compute_msd_at_radius(radius_init - delta)
    fd_grad = (loss_plus - loss_minus) / (2 * delta)
    
    print(f"Finite Difference Grad: {fd_grad:.4f}")
    
    rel_error = abs(grad_val - fd_grad) / abs(fd_grad)
    print(f"Relative Error: {rel_error:.4f}")
    
    if rel_error < 0.1:
        print("SUCCESS: Analytical gradient matches finite difference.")
    else:
        print("WARNING: Gradient mismatch.")
