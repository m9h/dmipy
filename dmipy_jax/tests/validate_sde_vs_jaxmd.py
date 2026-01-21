import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
from dmipy_jax.simulation.sde_models import RestrictedAnalyticSDE, solve_restricted_sde_batch
from dmipy_jax.core.particle_engine import brownian_step, non_periodic_box

def validate_sde_vs_jaxmd():
    print("Initializing Stochastic Trajectories Validation...")
    
    # Physics Parameters
    k_trap = 2.0  # Stiffness (1/s)
    D = 1.5       # Diffusivity (um^2/s)
    dim = 3
    center = jnp.zeros(dim)
    
    # Simulation Parameters
    N_particles = 4000
    total_time = 2.0 # seconds
    dt = 1e-3
    t_eval = jnp.linspace(0, total_time, 50)
    
    key = jax.random.PRNGKey(42)
    key_sde, key_md = jax.random.split(key)
    
    # --- 1. Diffrax SDE Solution ---
    print("Running Diffrax SDE Simulation...")
    sde_model = RestrictedAnalyticSDE(k=k_trap, diffusivity=D, center=center, dim=dim)
    y0 = jnp.zeros((N_particles, dim))
    
    save_at = diffrax.SaveAt(ts=t_eval)
    
    sol_sde = solve_restricted_sde_batch(
        sde_model, 
        (0.0, total_time), 
        y0, 
        dt0=dt, 
        key=key_sde,
        save_at=save_at
    )
    # sol_sde.ys shape: (N_particles, n_times, dim)
    # MSD calculation
    pos_sde = sol_sde.ys
    msd_sde = jnp.mean(jnp.sum(pos_sde**2, axis=-1), axis=0) # (n_times,)
    
    # --- 2. Manual JAX-MD (Langevin) Solution ---
    print("Running Manual Langevin Simulation...")
    
    # Langevin Step:
    # dX = -k*X*dt + sqrt(2*D*dt)*dW
    
    # Initialize
    # We use scan for speed
    
    def step_fn(carry, t):
        pos, key = carry
        current_k = k_trap
        
        # 1. Deterministic Drift (Euler)
        force = -current_k * (pos - center)
        drift = force * dt
        
        # 2. Stochastic Diffusion
        key, step_key = jax.random.split(key)
        # brownian_step expects diffusion_coeff. 
        # It handles dX = sqrt(2*D*dt)*noise.  
        # Wait, brownian_step in particle_engine takes dt, looks like:
        # scale = sqrt(2 * diffusion_coeff * dt)
        # dr = noise * scale
        # So we can just use it directly.
        
        # Note: shift_fn doesn't matter for infinite box
        shift_fn = non_periodic_box()[1] # returns (R, dR) -> R+dR
        pos_stoch = brownian_step(step_key, pos, shift_fn, diffusion_coeff=D, dt=dt)
        
        # Combine: new_pos = pos + drift + (pos_stoch - pos)
        # Ideally we do this carefully. 
        # brownian_step returns P_new = P_old + dX_stoch.
        # We need P_final = P_old + dX_drift + dX_stoch
        # So P_final = pos_stoch + drift
        
        pos_new = pos_stoch + drift
        
        return (pos_new, key), pos_new

    # Initial state
    init_state = (y0, key_md)
    
    # We need to scan over time steps matching t_eval resolution
    # To keep it simple, we simulate every dt but save only at t_eval indices (approx).
    # Or just save commonly if dt is small enough.
    
    # Let's just run scan over all steps
    n_steps = int(total_time / dt)
    ts = jnp.arange(n_steps) * dt
    
    _, trajectory_full = jax.lax.scan(step_fn, init_state, ts)
    # trajectory_full shape: (n_steps, N, dim)
    
    # Compute MSD for manual
    msd_md_full = jnp.mean(jnp.sum(trajectory_full**2, axis=-1), axis=1)
    
    # Interpolate to t_eval points for comparison
    # indices
    t_full = jnp.arange(n_steps) * dt
    msd_md = jnp.interp(t_eval, t_full, msd_md_full)
    
    # --- 3. Analytical Solution ---
    # MSD(t) = (dim * D / k) * (1 - exp(-2 * k * t))
    # Note: This assumes starting at x=0
    msd_theory = (dim * D / k_trap) * (1 - jnp.exp(-2 * k_trap * t_eval))
    
    # --- Comparison ---
    print("\nResults:")
    print(f"{'Time(s)':<10} | {'Theory':<10} | {'Diffrax':<10} | {'Manual':<10}")
    print("-" * 50)
    
    indices = jnp.linspace(0, len(t_eval)-1, 10, dtype=int)
    
    max_error_sde = 0.0
    max_error_md = 0.0
    
    for i in indices:
        t = t_eval[i]
        th = msd_theory[i]
        sde = msd_sde[i]
        md = msd_md[i]
        
        print(f"{t:<10.2f} | {th:<10.4f} | {sde:<10.4f} | {md:<10.4f}")
        
        if t > 0.05: # Skip t=0
            err_sde = abs(sde - th) / th
            err_md = abs(md - th) / th
            max_error_sde = max(max_error_sde, err_sde)
            max_error_md = max(max_error_md, err_md)
            
    print("-" * 50)
    print(f"Max Relative Error (Diffrax SDE): {max_error_sde:.4%}")
    print(f"Max Relative Error (Manual MD):   {max_error_md:.4%}")
    
    # Success Criteria
    # Comparing Stochastic methods requires tolerance. 
    # With 4000 particles, standard error is roughly 1/sqrt(4000) ~ 1.5%.
    # So < 5% error is good.
    
    if max_error_sde < 0.05 and max_error_md < 0.05:
        print("\n✅ PASSED: Both implementations match theory within statistical noise.")
    else:
        print("\n⚠️ WARNING: Deviation detected. Check statistics or implementation.")

if __name__ == "__main__":
    validate_sde_vs_jaxmd()
