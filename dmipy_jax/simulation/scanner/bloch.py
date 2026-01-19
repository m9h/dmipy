
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from functools import partial

# Gyromagnetic ratio for Hydrogen (Hz/T) - reduced/angular frequency form often used
# gamma_bar = 42.58 MHz/T
# gamma = 2 * pi * 42.58e6 rad/s/T
# We'll use the value from constants if available, or define it.
# The user prompt specifically asked for "cross product M x gamma * Bloc",
# implies Bloc is in Tesla and we need gamma in rad/s/T for correct units in dM/dt if t is seconds.
# Let's try to import from constants first or define it.
try:
    from dmipy_jax.constants import GYRO_MAGNETIC_RATIO
except ImportError:
    GYRO_MAGNETIC_RATIO = 2.6751525e8 # rad s^-1 T^-1

def bloch_dynamics(t, Magnetization, args):
    """
    Bloch equation dynamics.
    
    dM/dt = M x (gamma * B_loc) - R * (M - M_eq)
    
    Args:
        t: Time [s]
        Magnetization: Magnetization vector M(t) [Mx, My, Mz]
        args: Tuple containing:
            T1: Longitudinal relaxation time [s]
            T2: Transverse relaxation time [s]
            gradient_func: function G(t) -> [Gx, Gy, Gz] (T/m)
            B1_func: function B1(t) -> [B1x, B1y] (Tesla) (RF pulse)
            position: Position vector r [x, y, z] (m)
            delta_B_inhom: Field inhomogeneity (Tesla)
            
    Returns:
        dMdt: Rate of change of magnetization
    """
    T1, T2, gradient_func, B1_func, position, delta_B_inhom = args
    
    # 1. Get G(t) and B1(t)
    # Ensure they return jnp arrays
    G_t = gradient_func(t) # (3,)
    B1_t = B1_func(t)      # (2,) or (3,) - usually RF is transverse (x,y)
    
    # Handle B1 shape. If 2D (real/imag or x/y), assume z=0.
    # If B1 is complex scalar, convert to vector? Assuming B1_func returns vector for now.
    # Let's assume B1_func returns (B1x, B1y) or (B1x, B1y, 0).
    # If it returns a complex number, we'd need to handle that.
    # Given the prompt mentions "Get G(t) and B1(t)", we'll stick to vector math.
    
    # Pad B1 to 3D if needed
    # Pad B1 to 3D if needed
    B1_vec = jnp.zeros(3, dtype=B1_t.dtype)
    B1_vec = B1_vec.at[:B1_t.shape[0]].set(B1_t)
    
    # 2. Compute Bloc = B0 + (G . r) + delta_B_inhom
    # In rotating frame, B0 is effectively removed (or reduced to off-resonance).
    # "Bloc=B0+(G⋅r)+ΔBinhom" in the prompt likely refers to the effective Bz component
    # plus the RF field B1.
    # Usually B0 is along z.
    # Let's assume B0 term here is the *residual* B0 or off-resonance in rotating frame?
    # Or is this lab frame? standard Bloch implies lab frame if B0 is large.
    # But usually simulations are in rotating frame.
    # "Compute M x gamma Bloc"
    # If B1 is present, B_loc must include it.
    
    # Effective Bz from gradients and inhomogeneity
    B_z_eff = jnp.dot(G_t, position) + delta_B_inhom
    
    # Total B_loc vector
    # B_loc = B1_vec + [0, 0, B_z_eff]
    # (Assuming B1 is transverse)
    B_loc = B1_vec + jnp.array([0.0, 0.0, B_z_eff])
    
    # 3. Compute cross product M x gamma * Bloc
    # torque = gamma * (M x B_loc)
    torque = GYRO_MAGNETIC_RATIO * jnp.cross(Magnetization, B_loc)
    
    # 4. Relaxation
    # Longitudinal relax (z) -> (M0 - Mz)/T1
    # Transverse relax (x, y) -> -Mxy/T2
    # We assume M_eq is along z with magnitude 1 (or |M| if specified, but usually normalized).
    # Let's assume M0 = [0, 0, 1] relative to the initial magnitude.
    # Or simply:
    # dMx/dt = ... - Mx/T2
    # dMy/dt = ... - My/T2
    # dMz/dt = ... - (Mz - M0)/T1
    
    Mx, My, Mz = Magnetization
    M0 = 1.0 # Equilibrium magnetization magnitude
    
    relax_x = -Mx / T2
    relax_y = -My / T2
    relax_z = (M0 - Mz) / T1
    
    relaxation = jnp.array([relax_x, relax_y, relax_z])
    
    dMdt = torque + relaxation
    
    return dMdt

def _make_continuous_func(discrete_data, dt, duration):
    """
    Interpolates discrete waveform data to create a continuous function of time.
    """
    # Simple linear interpolation or just hold?
    # Diffrax likes smooth functions for higher order solvers.
    # Linear interpolation is usually fine.
    
    ts = jnp.linspace(0, duration, len(discrete_data))
    
    def func(t):
        return jnp.interp(t, ts, discrete_data, left=0.0, right=0.0) # Handle scaling/broadcasting if multi-dim?
        
    # jnp.interp works on 1D y. If discrete_data is (N, 3), we need to vmap the interp or handle it.
    # Let's assume discrete_data is shape (N, D).
    
    def vector_func(t):
        # vmap interp over columns
        # y: (N, D) -> need D interpolations
        # Transpose to (D, N) for vmapping over first axis?
        data_T = discrete_data.T # (D, N)
        res = jax.vmap(lambda d: jnp.interp(t, ts, d, left=0.0, right=0.0))(data_T)
        return res
        
    return vector_func

@eqx.filter_jit
def simulate_acquisition(phantom, sequence, duration, max_steps=10**7):
    """
    Simulates MRI acquisition.
    
    Args:
        phantom: Object with .positions (N, 3), .T1 (N,), .T2 (N,), .B0_inhom (N,)
        sequence: Object with .gradients (steps, 3), .rf (steps,), .dt (float)
                  (Assuming discrete waveforms for now, will interp)
        duration: Total duration (s)
        max_steps: Maximum number of solver steps (default: 10^7)
        
    Returns:
        Complex sum sum(Mx + iMy)
    """
    
    # Unpack Phantom
    # Support both struct-like and dictionary-like access if possible, or just assume attributes
    positions = phantom.positions # (N, 3)
    T1s = phantom.T1 # (N,)
    T2s = phantom.T2 # (N,)
    
    # Optional B0 inhomogeneity
    if hasattr(phantom, 'B0_inhom'):
        B0_inhom = phantom.B0_inhom
    else:
        B0_inhom = jnp.zeros_like(T1s)
        
    # Unpack Sequence
    # Assuming sequence has attributes for waveforms.
    # If they are functions already, great. If arrays, we wrap them.
    # Let's assume they are JAX arrays representing timepoints.
    
    # Infer dt if possible
    dt_seq = None
    if hasattr(sequence, 'time_points') and len(sequence.time_points) > 1:
        dt_seq = sequence.time_points[1] - sequence.time_points[0]
    elif hasattr(sequence, 'dt'):
        dt_seq = sequence.dt

    if callable(sequence.gradients):
        grad_func = sequence.gradients
    else:
        # Create interpolator
        # _make_continuous_func doesn't actually use 'dt' arg (it uses duration/len), 
        # but we kept the signature. Let's pass dt_seq or dummy.
        grad_func = _make_continuous_func(sequence.gradients, dt_seq, duration)
        
    if callable(getattr(sequence, 'rf', None)):
        B1_func = sequence.rf
    else:
        # Handle RF complex waveform -> B1 vec
        # Check if we have combined rf or split amp/phase
        if hasattr(sequence, 'rf'):
            rf_data = sequence.rf # (steps,) complex or (steps, 2)
        elif hasattr(sequence, 'rf_amplitude') and hasattr(sequence, 'rf_phase'):
            # Reconstruct complex
            rf_data = sequence.rf_amplitude * jnp.exp(1j * sequence.rf_phase)
        else:
             # Assume zero
             rf_data = jnp.zeros(len(sequence.time_points), dtype=jnp.complex64) if hasattr(sequence, 'time_points') else jnp.array([0j])

        # Helper to convert to (steps, 2) real
        if jnp.iscomplexobj(rf_data):
            rf_vec_data = jnp.stack([rf_data.real, rf_data.imag], axis=-1)
        elif rf_data.ndim == 1:
            # Assume strictly Bx? or magnitude?
            # Let's assume x-component
            rf_vec_data = jnp.stack([rf_data, jnp.zeros_like(rf_data)], axis=-1)
        else:
            rf_vec_data = rf_data # Assume (steps, 2)
            
        B1_func = _make_continuous_func(rf_vec_data, dt_seq, duration)

    # Initial Magnetization: All spins relaxed along Z
    # M_init = [0, 0, 1] for all N
    N = positions.shape[0]
    m0_single = jnp.array([0., 0., 1.])
    m_init = jnp.tile(m0_single, (N, 1)) # (N, 3)
    
    # Prepare Solver
    # We want to solve for all particles.
    # diffrax doesn't natively batch ODETerm over 'y' unless we treat y as a big PyTree or array.
    # If we pass y as (N, 3), bloch_dynamics needs to handle (N, 3).
    # But bloch_dynamics is written for single vector (3,).
    # We can vmap bloch_dynamics inside the term, or vmap the solver.
    # Usually easier to vmap the dynamics function so the state is just a large array to the solver.
    
    # args must match batched structure or be broadcasted.
    # T1, T2, position, delta_B_inhom are per-particle (N,).
    # grad_func, B1_func are scalar/global (shared).
    
    # We need a wrapper to handle the batching for the solver.
    # The solver sees state Y (N, 3).
    # It calls vector_field(t, Y, args).
    # We want vector_field to compute dY (N, 3).
    
    def batched_dynamics(t, M_batch, args_batch):
        # args_batch: (T1s, T2s, grad_func, B1_func, positions, B0_inhoms)
        # Note: grad_func and B1_func are NOT batched, they are functions.
        # T1s, T2s, positions, B0_inhoms ARE batched.
        
        T1s, T2s, grad_f, B1_f, pos_s, b0_s = args_batch
        
        # We need to map bloch_dynamics over the batch dimension of M and particle-specific args.
        # But we pass the SAME functions G(t) and B1(t) to all.
        
        # Define single-particle wrapper that takes just the varying args
        def single_particle_dy(m, t1, t2, pos, b0):
            # Reconstruct args tuple expected by bloch_dynamics
            single_args = (t1, t2, grad_f, B1_f, pos, b0)
            return bloch_dynamics(t, m, single_args)
            
        dM_dt_batch = jax.vmap(single_particle_dy)(M_batch, T1s, T2s, pos_s, b0_s)
        return dM_dt_batch

    term = diffrax.ODETerm(batched_dynamics)
    
    # Adaptive step size controller for speed
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    
    # Solver
    solver = diffrax.Dopri5() # Good default for adaptive
    
    # Time span
    t0 = 0.0
    t1 = duration
    dt0 = duration / 100.0 # Initial guess, controller will adjust
    if dt_seq is not None:
         dt0 = dt_seq # Better guess if available
         
    # Args
    solver_args = (T1s, T2s, grad_func, B1_func, positions, B0_inhom)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=m_init,
        args=solver_args,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps # Use argument
    )
    
    # Final state
    M_final = sol.ys[-1] # (N, 3)
    
    # Compute complex signal sum(Mx + iMy)
    Mx = M_final[:, 0]
    My = M_final[:, 1]
    
    signal = jnp.sum(Mx + 1j * My)
    
    return signal
