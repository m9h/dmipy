
import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.acquisition import JaxAcquisition
from jaxtyping import Array, Float

class ZeppelinNetwork(eqx.Module):
    """
    Amortized Inference Network for the Zeppelin Model.
    
    Predicts model parameters directly from the acquisition signal.
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, key, n_input_measurements, width_size: int = 64, depth: int = 3):
        """
        Args:
            key: JAX PRNGKey.
            n_input_measurements: Size of input signal (number of measurements).
            width_size: Width of hidden layers.
            depth: Number of hidden layers.
        """
        # Output size is 5: lambda_par, lambda_perp, fraction, theta, phi
        self.mlp = eqx.nn.MLP(
            in_size=n_input_measurements,
            out_size=5,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key
        )

    def __call__(self, signal: Float[Array, " N"]) -> dict:
        """
        Forward pass.
        
        Args:
            signal: Input signal array of shape (N_measurements,).
            
        Returns:
            Dictionary of predicted parameters:
                - lambda_par (diffusivity parallel)
                - lambda_perp (diffusivity perpendicular)
                - fraction (unused in pure Zeppelin but requested)
                - mu (orientation vector converted from theta, phi)
        """
        # Get raw MLP output
        raw_output = self.mlp(signal)
        
        # Split outputs
        # 0: lambda_par_raw
        # 1: lambda_perp_raw
        # 2: fraction_raw
        # 3: theta_raw
        # 4: phi_raw
        
        # Apply constraints
        lambda_par = jax.nn.softplus(raw_output[0])
        lambda_perp = jax.nn.softplus(raw_output[1])
        fraction = jax.nn.sigmoid(raw_output[2])
        
        # Angles (unconstrained raw output is fine, or we could bound them)
        theta = raw_output[3]
        phi = raw_output[4]
        
        return {
            'lambda_par': lambda_par,
            'lambda_perp': lambda_perp,
            'fraction': fraction,
            'mu': jnp.array([theta, phi])
        }

def self_supervised_loss(network: ZeppelinNetwork, data: Float[Array, " N"], acquisition: JaxAcquisition):
    """
    Self-supervised reconstruction loss.
    
    Args:
        network: The ZeppelinNetwork instance.
        data: Observed signal array (N_measurements,).
        acquisition: The acquisition scheme.
        
    Returns:
        MSE loss between data and reconstructed signal.
    """
    # 1. Predict parameters
    params = network(data)
    
    # 2. Reconstruct signal using Zeppelin model
    # Note: 'fraction' is predicted but ignored by the pure Zeppelin model as per plan.
    zeppelin_model = Zeppelin(
        lambda_par=params['lambda_par'],
        lambda_perp=params['lambda_perp'],
        mu=params['mu']
    )
    
    # 3. Generate signal
    # We unpack acquisition arrays
    s_recon = zeppelin_model(
        bvals=acquisition.bvalues,
        gradient_directions=acquisition.gradient_directions
    )
    
    # 4. Compute Loss (MSE)
    loss = jnp.mean((data - s_recon) ** 2)
    
    return loss
