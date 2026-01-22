
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Callable, Any
from jaxtyping import Array, Float

class GaussianPhaseApproximation:
    r"""
    Ground Truth Physics for training the Neural CDE.
    Computes signal for arbitrary waveform G(t) under Gaussian Diffusion assumption.
    
    S = S0 * exp( - \int_0^T q(t)^T D q(t) dt )  (Simplification of GPA)
    Using q(t) = gamma * \int G(t') dt'
    
    Actually, the full GPA implies b-value calculation:
    b = \int_0^T |q(t)|^2 dt  (trace of b-matrix for isotropic D)
    S = exp(-b * D)
    """
    @staticmethod
    def compute_q_trajectory(
        gradients: Float[Array, "T 3"], 
        dt: float, 
        gamma: float = 2.675e8 # Gyromagnetic ratio (rad/s/T)
    ) -> Float[Array, "T 3"]:
        """
        Integrates G(t) to get q(t).
        q(t) = gamma \int_0^t G(tau) dtau
        """
        # Cumulative sum ~ Integral
        # gradients shape (T, 3)
        # Result (T, 3) in m^-1
        # Use simple cumsum for checking.
        # q[i] = q[i-1] + gamma * G[i] * dt
        
        # NOTE: Gradients usually in T/m. Gamma in rad/s/T. q in rad/m.
        # Often normalized to 1/(2pi). Let's stick to SI units but arbitrary scaling for neural net learning.
        # CDE doesn't care about units, it learns the mapping.
        
        qt = jnp.cumsum(gradients, axis=0) * dt * gamma
        return qt

    @staticmethod
    def forward(
        gradients: Float[Array, "T 3"],
        dt: float,
        diffusivity: float,
        gamma: float = 1.0 # Normalized gamma
    ) -> Float[Array, ""]:
        """
        Computes GPA Signal.
        S = exp(- D * \int |q(t)|^2 dt )
        """
        qt = GaussianPhaseApproximation.compute_q_trajectory(gradients, dt, gamma)
        
        # q_squared shape (T, )
        q_squared = jnp.sum(qt**2, axis=-1)
        
        # Integral
        b_val = jnp.sum(q_squared) * dt
        
        return jnp.exp(-diffusivity * b_val)


class NeuralCDE(eqx.Module):
    r"""
    Neural Controlled Differential Equation for Waveform-Agnostic Signal Prediction.
    
    Dynamics:
        dz(t)/dt = f(z(t)) . dX(t)/dt
        
    Here, the control path X(t) is the q-trajectory (accumulated phase).
    dX(t)/dt is proportional to the Gradient G(t).
    
    So dz/dt = MLP(z) * G(t).
    """
    func: eqx.nn.MLP
    encoder: eqx.nn.Linear
    readout: eqx.nn.Linear
    hidden_dim: int
    
    def __init__(self, in_features=3, hidden_dim=32, out_features=1, key=None):
        """
        Args:
            in_features: 3 for 3D gradient vector.
            hidden_dim: Size of latent state z.
        """
        k1, k2, k3 = jax.random.split(key, 3)
        self.hidden_dim = hidden_dim
        
        # Vector Field f(z): Maps z -> Matrix[hidden, input]
        # Actually it's simpler: term = f(z) * dX.
        # If z is (H,), dX is (3,), we want output (H,).
        # So f(z) should output (H, 3).
        self.func = eqx.nn.MLP(
            in_size=hidden_dim, 
            out_size=hidden_dim * in_features, 
            width_size=64, 
            depth=2, 
            activation=jax.nn.tanh,
            key=k1
        )
        
        # Maps G(0)/Initial Cond to z0
        # Actually we usually start z=0. Or learned z0.
        # Let's map q(0) (which is 0) -> z0? No.
        # Standard: Learned constant or map from separate static features (e.g. tissue props).
        # Here we only input waveform. So z0 is fixed (e.g. 0) or learned param.
        # Let's use a learned encoding of the initial gradient? 
        # Usually G(0)=0.
        # Let's start with z0 = 0.
        
        # For simplicity, let's learn a linear map from empty to z0?
        # Or just use a fixed Projection of a dummy input.
        self.encoder = eqx.nn.Linear(1, hidden_dim, key=k2) 
        
        # Readout: z(T) -> Signal (Scalar)
        self.readout = eqx.nn.Linear(hidden_dim, out_features, key=k3)

    def __call__(self, times, gradients):
        """
        Args:
            times: (T,) array of time points.
            gradients: (T, 3) array of gradient waveforms.
        """
        # 1. Define Control Path X(t)
        # We want the CDE to be driven by G(t) or q(t)?
        # Physical intuition: G(t) drives the change.
        # So Path X(t) = Integral(G) = q(t).
        # diffrax.ControlTerm(vector_field, control) calculates vf(z) . d(control)/dt
        # d(control)/dt = G(t).
        # So using q(t) as control path naturally inputs G(t) into the eqn.
        
        # Integrate gradients to get path control points
        # Assuming uniform dt for input array, but LinearInterpolation handles t.
        # q ~ cumsum(G) * dt.
        # But wait, LinearInterpolation(ts, ys) takes the VALUES ys.
        # If we pass `control = LinearInterpolation(times, q_vals)`, then `control.evaluate(t)` returns q(t).
        # And `d(control)/dt` returns G(t).
        
        # Calculate q(t) roughly for the interpolation
        dt = times[1] - times[0]
        q_vals = jnp.cumsum(gradients, axis=0) * dt # Approximate integral sequence
        q_vals = jnp.concatenate([jnp.zeros((1, 3)), q_vals[:-1]]) # Shift? q(0)=0.
        # Let's just use LinearInterpolation on the provided `path_values`.
        # Actually simpler: Use gradients as path? Then dX/dt = derivative of gradient (slew).
        # Physics: Phase accumulates via G.
        # Let's use q_vals (Integral of G) as the path.
        
        control = diffrax.LinearInterpolation(times, q_vals)
        
        # 2. Define Term
        # term = ODETerm? No, ControlTerm.
        # dZ = f(Z) dX
        
        # Vector field function needs to match signature: (t, z, args) -> (H, 3)
        # Our MLP takes z -> (H*3). Reshape.
        def vector_field(t, z, args):
            out = self.func(z)
            return out.reshape(self.hidden_dim, 3)
            
        term = diffrax.ControlTerm(vector_field, control)
        solver = diffrax.Tsit5()
        
        # 3. Solve
        # z0 from encoder (dummy input 1.0)
        z0 = self.encoder(jnp.array([1.0]))
        
        sol = diffrax.diffeqsolve(
            term, 
            solver, 
            t0=times[0], 
            t1=times[-1], 
            dt0=dt, 
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )
        
        z_final = sol.ys[-1]
        
        # 4. Readout
        # Signal should be in [0, 1]. Use sigmoid or exp?
        # Diffusion is Decay. Exp(-ReLU(out))?
        # Or sigmoid.
        logit = self.readout(z_final)
        return jax.nn.sigmoid(logit) # Constrain to [0,1]

