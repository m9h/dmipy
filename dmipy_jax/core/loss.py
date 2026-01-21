import jax
import jax.numpy as jnp
from typing import Optional, Union

def rician_nll(observed: jnp.ndarray, predicted: jnp.ndarray, sigma: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Computes the Negative Log Likelihood for Rician distributed data.
    
    NLL ~ (M^2 + S^2 - 2MS)/2sigma^2 - log(I0(MS/sigma^2))
    """
    sigma_sq = sigma**2
    z = (observed * predicted) / sigma_sq
    
    # Base MSE-like term: (M - S)^2 / 2sigma^2 (approximation at high SNR)
    # Exact Rician exponent term is (M^2 + S^2) / 2sigma^2
    # But NLL = -log(P). P has exp( -(M^2+S^2)/2sig^2 ). 
    # So NLL has + (M^2+S^2)/2sig^2.
    
    term1 = (observed**2 + predicted**2) / (2 * sigma_sq)
    
    # Bessel term: - log( I0(z) )
    # Use localized exp scaled i0e: I0(z) = exp(z) * i0e(z)
    # log(I0) = z + log(i0e(z))
    
    term2 = -(z + jnp.log(jax.scipy.special.i0e(z)))
    
    return jnp.mean(term1 + term2)

def l1_regularization(coeffs: jnp.ndarray, lambda_reg: float) -> jnp.ndarray:
    """
    L1 Regularization term.
    """
    return lambda_reg * jnp.sum(jnp.abs(coeffs))

def prior_loss(fractions: jnp.ndarray, priors: jnp.ndarray, strength: float) -> jnp.ndarray:
    """
    Penalty for deviation from tissue priors.
    Loss = strength * sum( (fraction - prior)^2 )
    
    Args:
        fractions: (N_tissues,) array of estimated fractions.
        priors: (N_tissues,) array of prior fractions.
        strength: Regularization strength.
    """
    return strength * jnp.sum((fractions - priors)**2)
