
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from jaxtyping import Float, Array, PRNGKeyArray
from typing import Tuple, Callable

from dmipy_jax.simulation.differentiable_walker import DifferentiableWalker, solve_differentiable_walker_batch
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO # Expected to be accessible, usually gamma ~ 2.675e8 rad/s/T for protons

# Constants
# Gyromagnetic ratio for protons in rad / (s * T)
# If not in constants, define it here
try:
    from dmipy_jax.constants import GYRO_MAGNETIC_RATIO
except ImportError:
    GYRO_MAGNETIC_RATIO = 267.513e6 

def trapezoidal_gradient_waveform(
    t: Float[Array, "times"], 
    G_amp: Float[Array, ""], 
    direction: Float[Array, "3"], 
    delta: Float[Array, ""], 
    Delta: Float[Array, ""]
) -> Float[Array, "times 3"]:
    """
    Generates a PGSE gradient waveform G(t).
    Simple block pulses for now (rise time = 0 for simplicity, or small linear ramp if needed).
    Here we implement ideal rectangular pulses for standard PGSE.
    
    Waveform:
    0 to delta: +G
    delta to Delta: 0
    Delta to Delta+delta: -G (Refocussing pulse flips spin, or we flip gradient effective sign)
    
    Standard convention in PGSE signal eq: Phase = gamma * Integral (G(t) * x(t))
    If we stick to lab frame without 180 pulse logic manually, we represent the effective gradient.
    Effective Gradient:
    [0, delta]: +G
    [Delta, Delta+delta]: -G
    
    Args:
        t: Time points (N_t,)
        G_amp: Gradient amplitude (T/m)
        direction: (3,) Unit vector
        delta: Pulse duration (s)
        Delta: Pulse separation (s)
    """
    # Normalize direction just in case
    direction = direction / (jnp.linalg.norm(direction) + 1e-9)
    
    # Pulse 1: 0 <= t < delta
    p1 = (t >= 0.0) & (t < delta)
    
    # Pulse 2: Delta <= t < Delta + delta
    p2 = (t >= Delta) & (t < (Delta + delta))
    
    # Envelope: +1 during p1, -1 during p2, 0 otherwise
    envelope = jnp.where(p1, 1.0, jnp.where(p2, -1.0, 0.0))
    
    # G(t) shape (N_t, 3)
    G_t = envelope[:, None] * G_amp * direction[None, :]
    
    return G_t

def accumulate_phase(
    trajectories: Float[Array, "steps dim"], 
    gradient_waveform: Float[Array, "steps dim"], 
    dt: float
) -> Float[Array, ""]:
    """
    Computes total phase accrual phi = gamma * integral(G(t) . x(t) dt)
    via Trapezoidal integration or Riemann sum.
    
    Args:
        trajectories: (N_steps, 3) pos in meters
        gradient_waveform: (N_steps, 3) G in T/m
        dt: Time step in seconds
        
    Returns:
        Scalar phase (radians)
    """
    # Dot product at each time step: G(t) . x(t)
    # Shape: (N_steps,)
    interaction = jnp.sum(trajectories * gradient_waveform, axis=-1)
    
    # Integrate
    integral = jnp.sum(interaction) * dt
    
    phi = GYRO_MAGNETIC_RATIO * integral
    return phi

class EndToEndSimulator(eqx.Module):
    """
    Differentiable Simulator connecting Acquisition -> Walker -> Signal.
    """
    walker: DifferentiableWalker
    
    def __init__(self, walker: DifferentiableWalker):
        self.walker = walker
        
    def __call__(
        self, 
        G_amp: Float[Array, ""],
        delta: Float[Array, ""],
        Delta: Float[Array, ""],
        key: PRNGKeyArray,
        N_particles: int = 1000,
        dt: float = 1e-4
    ) -> Float[Array, ""]:
        """
        Simulate signal E = | < exp(i phi) > |
        
        Args:
            G_amp: Gradient strength (T/m)
            delta: Duration (s)
            Delta: Separation (s)
        """
        # 1. Setup Time
        # Simulate until end of second pulse: T_max = Delta + delta
        # Add a small buffer safely
        T_max = Delta + delta + 1e-4 
        t_eval = jnp.arange(0, T_max, dt)
        
        # 2. Trajectories
        # Initial positions - for sphere/cylinder uniform sampling is ideal.
        # For Restricted diffusion inside sphere, current Walker assumes starting at 0 is valid?
        # NO, Walker takes y0. We should sample y0 uniformly inside the sphere.
        # For differentiability, we sample in Unit Sphere and scale by R.
        # Sampling uniformly in unit sphere:
        # Rejection sampling or analytical? 
        # Analytical: Random direction, Random radius^(1/3).
        
        rng_y0, rng_walk = jax.random.split(key)
        
        # Sample Unit Sphere Positions
        # Shape (N, 3)
        # Random directions (Gaussian / Norm)
        raw_gauss = jax.random.normal(rng_y0, (N_particles, 3))
        norms = jnp.linalg.norm(raw_gauss, axis=1, keepdims=True)
        directions = raw_gauss / norms
        
        # Random radii: U^(1/3) for 3D uniform volume
        # U ~ Uniform(0, 1)
        u_rad = jax.random.uniform(rng_y0, (N_particles, 1))
        radii_samp = u_rad ** (1.0/3.0)
        
        y0_unit = directions * radii_samp
        # Scale to physical R done inside Walker? 
        # No, Walker takes y0_physical.
        y0_phys = y0_unit * self.walker.radius
        
        trajectories = solve_differentiable_walker_batch(
            self.walker, 
            (0.0, T_max), 
            y0_phys, 
            dt, 
            rng_walk
        )
        # trajectories shape: (N_particles, N_steps, 3)
        
        # 3. Waveform
        # Direction: Fixed along X for simplicity, or Z.
        grad_dir = jnp.array([1.0, 0.0, 0.0]) 
        
        # Must match timestamps of trajectories.
        # Walker output is on dt steps? 
        # The walker loop uses dt. The number of steps is (T1-T0)/dt.
        # We need to construct G_t for the exact same steps.
        # The scan output has length num_steps.
        
        # Construct time vector matching scan output
        # num_steps from Walker
        num_steps = trajectories.shape[1]
        t_sim = jnp.linspace(0.0, T_max, num_steps)
        
        G_t = trapezoidal_gradient_waveform(t_sim, G_amp, grad_dir, delta, Delta)
        
        # 4. Phase & Signal
        # vmap phase accumulation over particles
        phases = jax.vmap(accumulate_phase, in_axes=(0, None, None))(trajectories, G_t, dt)
        
        # Signal = | Mean( exp(i phi) ) |
        signal_complex = jnp.mean(jnp.exp(1j * phases))
        signal_mag = jnp.abs(signal_complex)
        
        return signal_mag

if __name__ == "__main__":
    print("Running Differentiable MRI Simulator Verification...")
    
    # Setup
    R_true = 5.0e-6 # 5 microns
    D_0 = 2.0e-9    # 2 um^2/ms -> SI: 2e-9 m^2/s
    
    # Walker
    walker = DifferentiableWalker(diffusivity=D_0, radius=R_true, geometry_type='sphere')
    sim = EndToEndSimulator(walker)
    
    # Optimization Target:
    # Find Delta that maximizes sensitivity to Radius.
    # Sensitivity = | d(Signal) / d(Radius) |
    # We want to maximize this.
    
    # Fixed params
    Delta_val = 20e-3 # 20 ms
    delta_val = 10e-3 # 10 ms
    
    def objective_sensitivity(G_est):
        # Compute gradient of Signal w.r.t Radius at this G_amp
        def signal_closure(r):
            # Re-instantiate walker with dynamic r
            w_dyn = DifferentiableWalker(diffusivity=D_0, radius=r, geometry_type='sphere')
            s_dyn = EndToEndSimulator(w_dyn)
            key = jax.random.PRNGKey(42) 
            return s_dyn(G_amp=G_est, delta=delta_val, Delta=Delta_val, key=key, N_particles=500, dt=1e-4)
            
        grad_r = jax.grad(signal_closure)(R_true)
        # We want to maximize sensitivity magnitude, i.e. minimize negative squared grad
        return - (grad_r ** 2)

    print("computing sensitivity for G = 10 mT/m...")
    sens_low = objective_sensitivity(0.01)
    print(f"Sensitivity (sq grad) at 10mT/m: {-sens_low:.2e}")
    
    print("computing sensitivity for G = 80 mT/m...")
    sens_high = objective_sensitivity(0.08)
    print(f"Sensitivity (sq grad) at 80mT/m: {-sens_high:.2e}")
    
    # Optimization Loop (Simple Gradient Descent on G_amp)
    # Expectation: Higher G usually gives higher sensitivity to restriction (more signal attenuation).
    # But too high G might kill the signal completely (noise floor). 
    # Here noise is 0, so higher G -> more sensitivity until wrapping?
    
    grad_obj = jax.grad(objective_sensitivity)
    G_curr = 0.01 # Start low
    lr = 1e-11 # Learning rate for G (T/m) - gradients are huge (~1e9)
    
    print("\nOptimizing Gradient Strength G_amp to maximize sensitivity to Radius...")
    for i in range(10):
        loss = objective_sensitivity(G_curr)
        grad = grad_obj(G_curr)
        G_curr = G_curr - lr * grad 
        print(f"Iter {i+1}: G = {G_curr*1000:.1f} mT/m, neg_Sens = {loss:.2e}, Grad = {grad:.2e}")
        
    print("\nOptimization complete.")
    if -loss > -sens_low:
        print("SUCCESS: Optimized G_amp has higher sensitivity than initial.")
    else:
        print("WARNING: Optimization did not improve sensitivity.")
