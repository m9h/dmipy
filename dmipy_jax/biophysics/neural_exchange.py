import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Callable, Optional, Union, Tuple

class NeuralExchangeRate(eqx.Module):
    """
    Learns a time-dependent exchange rate K(t) using an MLP.
    Effectively models k_ie(t) (intra -> extra).
    """
    mlp: eqx.nn.MLP
    
    def __init__(
        self,
        key: jax.random.PRNGKey,
        in_size: int = 1,  # e.g., time t
        out_size: int = 1, # e.g., exchange rate k_ie
        width_size: int = 32,
        depth: int = 2
    ):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key
        )

    def __call__(self, t: float) -> jnp.ndarray:
        """
        Returns the exchange rate k_ie at time t.
        Ensures positivity using softplus.
        """
        t_vec = jnp.atleast_1d(t)
        raw_output = self.mlp(t_vec)
        # Use Softplus to ensure positive exchange rate
        return jax.nn.softplus(raw_output)

class KargoModel(eqx.Module):
    """
    Two-compartment Macroscopic Kärger Exchange Model.
    State M = [M_intra, M_extra]^T (Complex).
    
    ODE: dM/dt = -R*M + K(t)*M - i*omega(t)*M
    
    where:
      - R: Relaxation matrix (diagonal)
      - K(t): Exchange matrix [-k_ie, k_ei; k_ie, -k_ei]
      - omega(t): "Frequency" matrix. 
        For standard Kärger diffusion, we set omega(t) = -i * q(t)^2 * D.
        Thus -i * omega(t) = -q(t)^2 * D (Real decay).
    """
    exchange_network: NeuralExchangeRate
    D_intra: float
    D_extra: float
    f_intra: float
    R_intra: float
    R_extra: float

    def __init__(
        self,
        exchange_network: NeuralExchangeRate,
        D_intra: float,
        D_extra: float,
        f_intra: float,
        t2_intra: float = 1.0, 
        t2_extra: float = 1.0
    ):
        self.exchange_network = exchange_network
        self.D_intra = D_intra
        self.D_extra = D_extra
        self.f_intra = f_intra
        self.R_intra = 1.0 / t2_intra
        self.R_extra = 1.0 / t2_extra

    def __call__(self, t: float, M: jnp.ndarray, args: dict) -> jnp.ndarray:
        """
        Differential equation derivative dM/dt.
        """
        omega_func = args['omega_func']
        
        # 1. Relaxation: -R * M
        relaxation_term = jnp.array([-self.R_intra * M[0], -self.R_extra * M[1]])
        
        # 2. Exchange: K(t) * M
        # k_ie defined by network
        k_ie = self.exchange_network(t)[0]
        
        # Detailed balance: f_i * k_ie = f_e * k_ei
        f_extra = 1.0 - self.f_intra
        k_ei = k_ie * (self.f_intra / (f_extra + 1e-9))
        
        # flow_i_to_e = k_ie * M_i
        # flow_e_to_i = k_ei * M_e
        # dMi = - flow_i_to_e + flow_e_to_i
        # dMe = + flow_i_to_e - flow_e_to_i
        
        exchange_flow = k_ei * M[1] - k_ie * M[0]
        exchange_term = jnp.array([exchange_flow, -exchange_flow])
        
        # 3. "Frequency" / Diffusion Damping: -i * omega(t) * M
        # omega_func(t) must return vector [omega_intra, omega_extra]
        omega = omega_func(t)
        damping_term = -1j * omega * M
        
        dMdt = relaxation_term + exchange_term + damping_term
        return dMdt

def get_pgse_omega_func(G: float, delta: float, Delta: float, D_intra: float, D_extra: float):
    """
    Constructs omega(t) for PGSE sequence such that -i*omega(t) generates Kärger diffusion decay.
    
    q(t) = gamma * integral_0^t G(tau) dtau
    decay_rate = q(t)^2 * D
    We want term = -decay_rate * M
    ODE term is -i * omega * M
    So omega = -i * decay_rate = -i * q(t)^2 * D
    """
    gamma = 267.513e6 # rad/s/T
    
    # We use eqx.filter_jit or just standard functions. 
    # Use standard python closure, JAX will trace.
    
    def omega_func(t):
        # Calculate q(t)
        # 1. 0 < t < delta: q = gamma * G * t
        # 2. delta < t < Delta: q = gamma * G * delta (constant)
        # 3. Delta < t < Delta+delta: q = gamma * G * delta - gamma * G * (t-Delta)
        # 4. t > Delta+delta: q = 0
        
        q_0 = gamma * G * t
        q_plat = gamma * G * delta
        q_rew = gamma * G * delta - gamma * G * (t - Delta)
        
        # We can use jnp.where or select
        cond1 = t <= delta
        cond2 = (t > delta) & (t <= Delta)
        cond3 = (t > Delta) & (t <= Delta + delta)
        
        q = jnp.where(
            cond1, q_0,
            jnp.where(
                cond2, q_plat,
                jnp.where(cond3, q_rew, 0.0)
            )
        )
        
        # omega = -i * q^2 * D
        om_intra = -1j * (q**2) * D_intra
        om_extra = -1j * (q**2) * D_extra
        return jnp.array([om_intra, om_extra])

    return omega_func

def simulate_kargo_signal(
    model: KargoModel,
    bvals: jnp.ndarray,
    TE: float,
    delta: float = 0.01,
    Delta: float = 0.02
) -> jnp.ndarray:
    """
    Simulates Kargo model for a batch of b-values (via vmap).
    """
    # Verify bvals shape
    bvals = jnp.atleast_1d(bvals)
    
    # Pre-calculate G for each b-value
    # b = (gamma G delta)^2 (Delta - delta/3)
    gamma = 267.513e6
    b_si = bvals * 1e6 # s/m^2
    G_vals = jnp.sqrt(b_si / (gamma**2 * delta**2 * (Delta - delta/3)))
    
    # Simulation function for one G
    def sim_one(G):
        # Define omega func for this G
        def omega_func(t):
             # q(t) logic duplicated here to be safe and traced with G
             # ...
             q_0 = gamma * G * t
             q_plat = gamma * G * delta
             q_rew = gamma * G * delta - gamma * G * (t - Delta)
             
             cond1 = t <= delta
             cond2 = (t > delta) & (t <= Delta)
             cond3 = (t > Delta) & (t <= Delta + delta)
             
             q = jnp.where(
                 cond1, q_0,
                 jnp.where(
                     cond2, q_plat,
                     jnp.where(cond3, q_rew, 0.0)
                 )
             )
             
             om_i = -1j * (q**2) * model.D_intra
             om_e = -1j * (q**2) * model.D_extra
             return jnp.array([om_i, om_e])
        
        # Initial State
        m0 = jnp.array([model.f_intra, 1.0 - model.f_intra], dtype=jnp.complex128)
        
        # Solver
        term = diffrax.ODETerm(model)
        solver = diffrax.Tsit5() # Efficient for non-stiff. If K is large, maybe Kvaerno3.
        saveat = diffrax.SaveAt(ts=jnp.array([TE]))
        
        sol = diffrax.diffeqsolve(
            term, 
            solver, 
            t0=0.0, 
            t1=TE, 
            dt0=TE/50.0, 
            y0=m0, 
            args={'omega_func': omega_func},
            saveat=saveat,
            max_steps=4096
        )
        
        final_M = sol.ys[0] 
        return jnp.abs(jnp.sum(final_M))

    # Vectorize over G_vals
    signals = jax.vmap(sim_one)(G_vals)
    return signals
