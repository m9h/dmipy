import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Array, Float, PyTree
from typing import Callable, Dict, Union
from dmipy_jax.acquisition import JaxAcquisition

class AcquisitionProtocol(eqx.Module):
    """
    A learnable acquisition protocol.
    
    Parameters are stored in unconstrained space (logits) and transformed
    to physical values in the forward pass.
    """
    b_values_logits: Float[Array, "n_measurements"]
    directions_logits: Float[Array, "n_measurements 3"]
    TE_logits: Float[Array, "n_measurements"]
    
    max_b_value: float = eqx.field(static=True)
    min_TE: float = eqx.field(static=True)
    max_TE: float = eqx.field(static=True)
    
    def __init__(
        self, 
        n_measurements: int, 
        max_b_value: float = 3000.0, 
        min_TE: float = 0.05, 
        max_TE: float = 0.15, 
        key: jax.Array = None
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        keys = jax.random.split(key, 3)
        self.max_b_value = max_b_value
        self.min_TE = min_TE
        self.max_TE = max_TE
        
        # Initialize logits
        # b-values initialized to favor lower values initially (negative mean)
        self.b_values_logits = jax.random.normal(keys[0], (n_measurements,)) - 1.0
        # directions initialized randomly on unit sphere
        self.directions_logits = jax.random.normal(keys[1], (n_measurements, 3))
        # TE initialized to be small (negative mean)
        self.TE_logits = jax.random.normal(keys[2], (n_measurements,)) - 2.0

    def __call__(self) -> JaxAcquisition:
        # Sigmoid mapping for b-values [0, max_b]
        b_values = jax.nn.sigmoid(self.b_values_logits) * self.max_b_value
        
        # Normalize directions to unit vectors
        # Handle case where norm is near zero by adding epsilon
        norms = jnp.linalg.norm(self.directions_logits, axis=-1, keepdims=True)
        directions = self.directions_logits / (norms + 1e-6)
        
        # Sigmoid mapping for TE [min_TE, max_TE]
        TE = self.min_TE + jax.nn.sigmoid(self.TE_logits) * (self.max_TE - self.min_TE)
        
        # Derive timing parameters based on TE
        # Assume a standard valid timing relation for PGSE:
        # TE > Delta + delta
        # Let's assume constant small_delta for simplicity, or make it learnable.
        # Here we assume small_delta is fixed to a realistic value, and BigDelta is
        # derived from TE minus overhead.
        
        # Fixed small_delta (e.g., 10ms)
        delta = jnp.full_like(b_values, 0.010)
        
        # BigDelta set to TE/2 (approximate) or TE - small_delta - constant overhead
        # Let's simple assumption for demonstration: Delta = TE - 0.02 (20ms overhead)
        # Ensure Delta > delta
        Delta = jnp.maximum(TE - 0.02, delta + 0.005)

        return JaxAcquisition(
            bvalues=b_values,
            gradient_directions=directions,
            echo_time=TE,
            delta=delta,
            Delta=Delta
        )

class CramerRaoBound(eqx.Module):
    """
    Computes the Cramer-Rao Lower Bound (CRLB) for a given tissue model and protocol.
    """
    tissue_model: eqx.Module
    sigma: float = 1.0 # Noise standard deviation
    
    def __call__(self, tissue_params: Dict[str, float], protocol: JaxAcquisition) -> Float[Array, "n_params"]:
        """
        Calculate the diagonal of the CRB matrix (variance bounds).
        """
        
        # We need a function that takes ONLY the values of tissue_params as inputs
        # because jax.jacfwd needs to differentiate w.r.t them.
        # Equinox models usually take kwargs.
        
        param_keys = list(tissue_params.keys())
        param_values = jnp.array([tissue_params[k] for k in param_keys])

        def simulate_signal_flat(flat_params):
            # Reconstruct kwargs
            params_dict = {k: v for k, v in zip(param_keys, flat_params)}
            # Call model
            # We assume model returns signal array of shape (N,)
            return self.tissue_model(
                bvals=protocol.bvalues,
                gradient_directions=protocol.gradient_directions,
                big_delta=protocol.Delta,
                small_delta=protocol.delta, 
                tau=protocol.Delta - protocol.delta/3.0,
                **params_dict
            )
        
        # Jacobian: (N_measurements, N_params)
        J = jax.jacfwd(simulate_signal_flat)(param_values)
        
        # Fisher Information Matrix: J^T @ J / sigma^2
        FIM = (1.0 / self.sigma**2) * (J.T @ J)
        
        # Invert FIM
        # Add small epsilon to diagonal for numerical stability (regularization)
        FIM_reg = FIM + jnp.eye(FIM.shape[0]) * 1e-9
        CRB_matrix = jnp.linalg.inv(FIM_reg)
        
        return jnp.diag(CRB_matrix)

def optimize_acquisition(
    tissue_model: eqx.Module,
    target_params: Dict[str, float],
    n_measurements: int = 30,
    max_b_value: float = 3000.0,
    seed: int = 0
):
    """
    Optimizes an acquisition protocol to minimize the mean log CRB 
    for the given target parameters.
    """
    
    # Initialize protocol wrapper
    protocol_model = AcquisitionProtocol(
        n_measurements=n_measurements,
        max_b_value=max_b_value,
        key=jax.random.PRNGKey(seed)
    )
    
    # Define loss function
    def loss_fn(protocol_model, args):
        tissue_model_static, target_params_static = args
        
        # Generate scheme from current learnable parameters
        scheme = protocol_model()
        
        # Calculate CRB
        crb_calc = CramerRaoBound(tissue_model_static)
        variances = crb_calc(target_params_static, scheme)
        
        # A-optimality (trace of CRB) or D-optimality (log det FIM)
        # Here we use mean of log variances (minimizing geometric mean of variances)
        # It's a robust choice.
        loss = jnp.mean(jnp.log(variances))
        
        # Optional: Regularize for shorter TE to improve SNR? 
        # But CRB already accounts for noise sigma. If sigma depends on TE...
        # For now, constant sigma.
        
        return loss

    # Optimistix solver
    # Use Gradient Descent as it is more robust for this non-convex landscape with sigmoid mappings
    solver = optx.GradientDescent(learning_rate=0.05, rtol=1e-3, atol=1e-3)
    
    # Run minimization
    sol = optx.minimise(
        loss_fn,
        solver,
        protocol_model,
        args=(tissue_model, target_params),
        max_steps=500
    )
    
    return sol.value
