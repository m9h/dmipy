import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from typing import Optional, Tuple, Any, Callable

class RestrictedAnalyticSDE(eqx.Module):
    """
    Restricted diffusion using an Ornstein-Uhlenbeck (OU) process.
    Drift: -k * (x - center)
    Diffusion: Constant noise (Brownian motion)
    """
    k: float
    center: jnp.ndarray
    diffusivity: float

    def __init__(self, k: float, diffusivity: float, center: Optional[jnp.ndarray] = None, dim: int = 3):
        self.k = k
        self.diffusivity = diffusivity
        self.center = center if center is not None else jnp.zeros(dim)

    def drift(self, t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Drift term: -k * (y - center)
        """
        return -self.k * (y - self.center)

    def diffusion(self, t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Diffusion term: sqrt(2 * D) * I
        Returns a matrix of shape (dim, dim) if using standard Brownian motion of shape (dim,).
        """
        dim = y.shape[0]
        sigma = jnp.sqrt(2 * self.diffusivity)
        return jnp.eye(dim) * sigma

def solve_restricted_sde_batch(
    sde_model: RestrictedAnalyticSDE,
    t_span: Tuple[float, float],
    y0_batch: jnp.ndarray,
    dt0: float = 1e-3,
    key: Optional[jax.random.PRNGKey] = None,
    save_at: Optional[diffrax.SaveAt] = None
):
    """
    Simulates a batch of particles using the RestrictedAnalyticSDE model.
    
    Args:
        sde_model: Instance of RestrictedAnalyticSDE.
        t_span: Tuple (t0, t1).
        y0_batch: Initial positions of shape (N, dim).
        dt0: Initial step size.
        key: Random key.
        save_at: Checkpoints.
        
    Returns:
        Solution object with batched dimensions.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    N, dim = y0_batch.shape
    
    # Define the solve function for a single particle
    def solve_single(y0, single_key):
        brownian_motion = diffrax.VirtualBrownianTree(
            t_span[0], t_span[1], tol=1e-3, shape=(dim,), key=single_key
        )
        
        drift_term = diffrax.ODETerm(sde_model.drift)
        diffusion_term = diffrax.ControlTerm(sde_model.diffusion, brownian_motion)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        
        solver = diffrax.EulerHeun()
        
        # Default save_at if None
        save = save_at if save_at is not None else diffrax.SaveAt(ts=jnp.linspace(t_span[0], t_span[1], 100))
        
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt0,
            y0=y0,
            saveat=save,
            max_steps=100000 # Increased for safety
        )
        return sol

    # Split keys for the batch
    keys = jax.random.split(key, N)
    
    # Vmap the solver
    batch_sol = jax.vmap(solve_single)(y0_batch, keys)
    
    return batch_sol

if __name__ == "__main__":
    # verification script
    import matplotlib.pyplot as plt
    
    print("Running verification for RestrictedAnalyticSDE...")
    
    # Parameters
    k_init = 1.0  # Stiffness
    D = 1.0  # Diffusivity
    dim = 3
    N_particles = 5000
    T_max = 5.0
    
    # Define a function to calculate final MSD given k
    def get_final_msd(k_val):
        model = RestrictedAnalyticSDE(k=k_val, diffusivity=D, dim=dim)
        # Use a fixed key for differentiability check stability
        key = jax.random.PRNGKey(42)
        y0_batch = jnp.zeros((N_particles, dim))
        t_eval = jnp.linspace(0, T_max, 50)
        save_at = diffrax.SaveAt(ts=t_eval)
        
        sol = solve_restricted_sde_batch(model, (0, T_max), y0_batch, dt0=0.01, key=key, save_at= save_at)
        trajectories = sol.ys
        sq_displacement = jnp.sum(trajectories**2, axis=-1)
        msd = jnp.mean(sq_displacement, axis=0)
        return msd[-1], msd
        
    print("Simulating and checking gradient...")
    
    # Function to just get the scalar output for grad
    def loss(k_val):
        final_msd, _ = get_final_msd(k_val)
        return final_msd
        
    # JIT compilation
    loss_and_grad = jax.jit(jax.value_and_grad(loss))
    
    val, grad = loss_and_grad(k_init)
    
    final_msd_val, msd_curve = get_final_msd(k_init)
    t_eval = jnp.linspace(0, T_max, 50)
    
    # Theoretical MSD Saturation = 3D / k
    # d(MSD)/dk = -3D / k^2
    theory_val = 3 * D / k_init
    theory_grad = -3 * D / (k_init**2)
    
    print(f"Final MSD: {val:.4f} (Theory: {theory_val:.4f})")
    print(f"Gradient w.r.t k: {grad:.4f} (Theory: {theory_grad:.4f})")
    
    # Check saturation slope
    slope = (msd_curve[-1] - msd_curve[-5]) / (t_eval[-1] - t_eval[-5])
    print(f"Slope at end: {slope:.4f}")
    
    # Simple ASCII plot
    print("\nMSD vs Time:")
    msd_theory_curve = (dim * D / k_init) * (1 - jnp.exp(-2 * k_init * t_eval))
    for t, v, th in zip(t_eval[::5], msd_curve[::5], msd_theory_curve[::5]):
        bar_len = int(v * 10)
        print(f"t={t:.1f}: {'#' * bar_len} ({v:.2f}) vs Theory ({th:.2f})")
        
    if jnp.abs(val - theory_val) < 0.2 and jnp.abs(grad - theory_grad) < 0.5:
         print("\nSUCCESS: MSD saturates and implementation is differentiable.")
    else:
         print("\nWARNING: Verification metrics not close enough to theory.")
