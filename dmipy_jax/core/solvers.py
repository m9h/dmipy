import jax
import jax.numpy as jnp
import diffrax
from typing import Callable, Optional, Union, Tuple, Any
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO

class BlochSimulator:
    """
    Simulates the Bloch equations for Nuclear Magnetic Resonance.
    
    dM/dt = M x gamma B - R(M - M0)
    
    where:
    M is the magnetization vector [Mx, My, Mz]
    B is the magnetic field vector [Bx, By, Bz]
    gamma is the gyromagnetic ratio
    R is the relaxation matrix (diagonal with 1/T2, 1/T2, 1/T1)
    M0 is the equilibrium magnetization
    """
    
    def __init__(self, t1: float, t2: float, m0: Optional[jnp.ndarray] = None):
        """
        Args:
            t1: Longitudinal relaxation time (seconds).
            t2: Transverse relaxation time (seconds).
            m0: Equilibrium magnetization vector (default: [0, 0, 1]).
        """
        self.t1 = t1
        self.t2 = t2
        self.m0 = m0 if m0 is not None else jnp.array([0.0, 0.0, 1.0])
        
        # Relaxation matrix diagonal
        self.rates = jnp.array([1.0/t2, 1.0/t2, 1.0/t1])

    def vector_field(self, t, m, args):
        """
        Rate of change dM/dt.
        
        Args:
            t: Time
            m: Magnetization vector M(t)
            args: Tuple containing (gradient_func, position)
                  gradient_func: G(t) -> array (3,) [T/m] or similar scaling
                  position: r -> array (3,) [m]
        """
        grad_func, pos = args
        
        # 1. Gradient contribution to B-field
        # We assume the main field B0 is along Z and is removed by rotating frame.
        # The gradients add a Z-component to the effective field in the rotating frame:
        # B_eff_z = G(t) . r
        # We assume G(t) is a vector (Gx, Gy, Gz) in T/m.
        G = grad_func(t)
        
        # B_z_eff (Tesla)
        b_z_eff = jnp.dot(G, pos)
        
        # Omega (rad/s) = gamma * B
        omega_z = GYRO_MAGNETIC_RATIO * b_z_eff
        
        # Precession: M x Omega
        # Omega = [0, 0, omega_z]
        # Cross product:
        # [My * omega_z - Mz * 0]
        # [Mz * 0 - Mx * omega_z]
        # [Mx * 0 - My * 0]
        # = [My * omega_z, -Mx * omega_z, 0]
        
        # Note: Standard definition dM/dt = M x gamma B -> (My*w, -Mx*w, 0)
        # matches right hand rule for positive gamma and B along Z.
        cross_term = jnp.array([
            m[1] * omega_z,
            -m[0] * omega_z,
            0.0
        ])
        
        # Relaxation: -R * (M - M0)
        relaxation_term = -self.rates * (m - self.m0)
        
        return cross_term + relaxation_term

    def __call__(self, 
                 t_span: Tuple[float, float], 
                 m_init: jnp.ndarray, 
                 gradient_waveform: Callable[[float], jnp.ndarray], 
                 position: jnp.ndarray,
                 dt0: float = 1e-4,
                 max_steps: int = 10000,
                 save_at: Optional[diffrax.SaveAt] = None):
        """
        Run the simulation.
        
        Args:
            t_span: (t_start, t_end)
            m_init: Initial magnetization vector.
            gradient_waveform: Function G(t) returning gradient vector.
            position: Particle position vector.
            dt0: Initial step size.
            max_steps: Maximum solver steps.
            save_at: Controller for saving output (default: save t1).
            
        Returns:
            Solution object from diffrax.
        """
        if save_at is None:
            save_at = diffrax.SaveAt(t1=True)
            
        term = diffrax.ODETerm(self.vector_field)
        solver = diffrax.Tsit5()
        
        # args passed to vector_field
        args = (gradient_waveform, position)
        
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt0,
            y0=m_init,
            args=args,
            saveat=save_at,
            max_steps=max_steps
        )
        return sol


def solve_diffusion_sde(
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    drift: Callable[[float, jnp.ndarray,  Any], jnp.ndarray],
    diffusion: Callable[[float, jnp.ndarray, Any], jnp.ndarray],
    dt0: float = 1e-3,
    key: Optional[jax.random.PRNGKey] = None,
    args: Any = None,
    save_at: Optional[diffrax.SaveAt] = None
):
    """
    Solve a Stochastic Differential Equation for diffusion.
    
    dY = drift(t, Y) dt + diffusion(t, Y) dW
    
    Args:
        t_span: (t_start, t_end)
        y0: Initial state (e.g. positions of particles), shape (N, D).
        drift: Function f(t, y, args) -> (N, D)
        diffusion: Function g(t, y, args) -> (N, D, BrownianDim) or (N, D) 
                   representing scale of noise.
        dt0: Step size.
        key: JAX PRNG Key for noise.
        args: Extra arguments for drift/diffusion functions.
        save_at: Checkpoints.
        
    Returns:
        diffrax solution.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
        
    # Standard Brownian Motion
    # If y0 is (N, 3), we need brownian path for each particle?
    # Usually diffrax handles batching if mapped, or we treat state as a large vector.
    # However, specialized BrownianPath is better.
    # Here we assume y0 is a single particle state (D,) or we map over it externally.
    # To support batching inside:
    # We can use VirtualBrownianTree with shape matching y0.
    
    brownian_shape = y0.shape
    brownian_motion = diffrax.VirtualBrownianTree(
        t_span[0], t_span[1], tol=1e-3, shape=brownian_shape, key=key
    )
    
    # Drift and Diffusion need to be compliant with diffrax.ODETerm / ControlTerm
    # We use MultiTerm for SDEs.
    # weak=True for EulerHeun usually?
    
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, brownian_motion)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    
    solver = diffrax.EulerHeun()
    
    if save_at is None:
        save_at = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=save_at
    )
    
    return sol
