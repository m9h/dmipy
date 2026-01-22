
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from typing import NamedTuple, Callable, Union, Tuple, Optional

# Constants
try:
    from dmipy_jax.constants import GYRO_MAGNETIC_RATIO
except ImportError:
    GYRO_MAGNETIC_RATIO = 2.6751525e8 # rad s^-1 T^-1

class BlochTorreyGeometry(eqx.Module):
    """
    Geometry parameters for Bloch-Torrey simulation.
    """
    T1: Union[float, jax.Array]
    T2: Union[float, jax.Array]
    D: Union[float, jax.Array] # Scalar or Tensor (3, 3)

    def __init__(self, T1, T2, D):
        self.T1 = T1
        self.T2 = T2
        self.D = D

def _validate_waveform(waveform, t, shape=(3,)):
    # Helper to check waveform return shape
    pass

@eqx.filter_jit
def simulate_signal(
    geometry: BlochTorreyGeometry,
    waveform: Callable[[float], jax.Array],
    duration: float,
    dt: float = None,
    M0: jax.Array = jnp.array([0., 0., 1.]),
    max_steps: int = 100000
) -> jax.Array:
    """
    Simulates the MRI signal using the Bloch-Torrey equations with diffrax.
    
    Solves the coupled system:
    dM/dt = M x (gamma * G(t) * r) ... (Handled via k-space approx for free diffusion)
    
    Actually, for free diffusion (Gaussian Phase Approx), we solve:
    dM_xy/dt = -i (gamma * G(t) * r) M_xy ... -> this is spatial.
    
    The prompt asks for "Define the Bloch-Torrey ODE".
    And "Diffusion Tensor D".
    
    Standard Bloch-Torrey in k-space (Torrey 1956):
    dM/dt = ... - k(t).T * D * k(t) * M
    
    where k(t) = integral(gamma * G(tau) dtau).
    
    So we solve for state Y = [M_x, M_y, M_z, k_x, k_y, k_z].
    
    State Y shape: (6,)
    
    Args:
        geometry: PyTree containing T1, T2, D.
        waveform: Function f(t) -> [Gx, Gy, Gz] (T/m).
        duration: Total simulation time.
        dt: Initial step size (optional).
        M0: Initial magnetization.
        
    Returns:
        M_final: Magnetization vector at t=duration.
    """
    
    # 1. Define ODE System
    # Y = [Mx, My, Mz, kx, ky, kz]
    
    def vector_field(t, y, args):
        M = y[:3]
        k = y[3:]
        
        # Unpack geometry
        T1 = geometry.T1
        T2 = geometry.T2
        D = geometry.D
        
        # Get Gradient at time t
        G = waveform(t) # (3,)
        
        # 1. k-space evolution: dk/dt = gamma * G
        dk_dt = GYRO_MAGNETIC_RATIO * G
        
        # 2. Magnetization evolution
        Mx, My, Mz = M
        
        # Relaxation
        # dMx/dt = -Mx/T2
        # dMy/dt = -My/T2
        # dMz/dt = -(Mz - 1)/T1 (Assuming M_eq = 1)
        
        dMx_relax = -Mx / T2
        dMy_relax = -My / T2
        dMz_relax = -(Mz - 1.0) / T1
        
        # Diffusion Damping (Bloch-Torrey in k-space)
        # Atenuation factor rate = k^T D k
        # Applies to transverse magnetization? 
        # Torrey 1956 derivation usually applies to the complex transverse M_xy.
        # If we treat Mx, My independently:
        # dA/dt = -bD(t) * A
        # bD(t) = k(t)^T D k(t)
        
        # Handle Scalar vs Tensor D
        if jnp.ndim(D) == 0:
            diff_loss = D * jnp.dot(k, k)
        else:
            diff_loss = jnp.dot(k, jnp.dot(D, k))
            
        dMx_diff = -diff_loss * Mx
        dMy_diff = -diff_loss * My
        # Diffusion damping typically affects transverse signal in this framework.
        # Does it affect Z? Usually NO for diffusion encoding (PGSE), but technically diffusion happens in 3D.
        # However, the "b-value" attenuation is on the phase-encoded signal.
        # Standard approach: Apply to M_xy.
        
        dMz_diff = 0.0 # Diffusion doesn't decay longitudinal magnetization in equivalent way here?
        # Actually, Torrey equation is vector equation: dM/dt = ... + div(D grad M).
        # In k-space (Fourier domain wrt position):
        # M(k, t) -> dM/dt = ... - k^T D k M
        # So it SHOULD apply to all components if Mz is also spatially distributed?
        # But usually we care about the *coherent* signal.
        # If we assume we are tracking the *ensemble average* (signal), then yes, diffusion causes dephasing -> decay.
        # But for Mz, it's just T1 recovery usually.
        # Let's apply to Mx, My (transverse) as that's the signal we measure.
        
        dMx_dt = dMx_relax + dMx_diff
        dMy_dt = dMy_relax + dMy_diff
        dMz_dt = dMz_relax + dMz_diff # Zero diffusion term for Z for now, typically correct for PGSE signal models.
        
        dM_dt = jnp.stack([dMx_dt, dMy_dt, dMz_dt])
        
        return jnp.concatenate([dM_dt, dk_dt])

    # 2. Setup Solver
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    
    # Initial State
    # M0 provided, k0 = 0
    k0 = jnp.zeros(3)
    y0 = jnp.concatenate([M0, k0])
    
    # Time span
    t0 = 0.0
    t1 = duration
    dt0 = dt if dt is not None else duration / 100.0
    
    # Step Controller
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)
    
    # Adjoint method for gradients
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        max_steps=max_steps
    )
    
    # Extract final M
    y_final = sol.ys[-1]
    M_final = y_final[:3]
    
    return M_final

@eqx.filter_jit
def simulate_acquisition(
    phantom: eqx.Module, # IsochromatPhantom
    sequence: eqx.Module, # PulseSequence
    duration: float,
    dt: float = None,
    max_steps: int = 100000
) -> jax.Array:
    """
    Simulates an acquisition for a phantom and sequence.
    Returns the complex signal (Mx + iMy) summed over all spins at t=duration.
    
    Args:
        phantom: IsochromatPhantom containing spins.
        sequence: PulseSequence providing gradients and RF.
        duration: Simulation duration.
        
    Returns:
        Complex signal (scalar).
    """
    
    # 1. Kernel for single spin
    def simulate_spin(pos, T1, T2, M0_val, df):
        # State: M = [Mx, My, Mz]
        # Initial: M0 along z? Usually M0 is magnitude of equilibrium Mz.
        # Assume start at equilibrium: [0, 0, M0]
        y0 = jnp.array([0., 0., M0_val])
        
        def vector_field(t, M, args):
            # M is (3,)
            Mx, My, Mz = M
            
            # Fields
            # Gradients G(t) in T/m
            G = sequence.get_gradients(t) # (3,)
            # RF B1(t) in Tesla (complex) -> B1x + iB1y
            B1 = sequence.get_rf(t)
            
            # Omega_z = gamma * (G . r + df/gamma * 2pi?) 
            # df is off-resonance in Hz. Omega = 2pi * df.
            bz_grad = jnp.dot(G, pos) # T
            omega_z = GYRO_MAGNETIC_RATIO * bz_grad + 2 * jnp.pi * df
            
            # Omega_xy
            omega_x = GYRO_MAGNETIC_RATIO * jnp.real(B1)
            omega_y = GYRO_MAGNETIC_RATIO * jnp.imag(B1)
            
            # dMx = My*Oz - Mz*Oy
            dMx_rot = My * omega_z - Mz * omega_y
            # dMy = Mz*Ox - Mx*Oz
            dMy_rot = Mz * omega_x - Mx * omega_z
            # dMz = Mx*Oy - My*Ox
            dMz_rot = Mx * omega_y - My * omega_x
            
            # Relaxation
            dMx_rel = -Mx / T2
            dMy_rel = -My / T2
            dMz_rel = -(Mz - M0_val) / T1
            
            return jnp.array([dMx_rot + dMx_rel, dMy_rot + dMy_rel, dMz_rot + dMz_rel])

        # ODE Solver
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-6)
        
        dt0 = dt if dt is not None else duration / 100.0
        
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=duration, dt0=dt0, y0=y0,
            stepsize_controller=stepsize_controller, max_steps=max_steps
        )
        return sol.ys[-1]

    # 2. Vmap over phantom
    # phantom.positions: (N, 3)
    # phantom.T1: (N,)
    
    sim_func = jax.vmap(simulate_spin)
    M_finals = sim_func(phantom.positions, phantom.T1, phantom.T2, phantom.M0, phantom.off_resonance)
    
    # 3. Aggregate Signal
    # Signal = Sum(Mx + iMy)
    Mx = M_finals[:, 0]
    My = M_finals[:, 1]
    
    signal = jnp.sum(Mx + 1j * My)
    return signal
