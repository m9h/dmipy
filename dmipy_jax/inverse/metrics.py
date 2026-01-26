
import jax
import jax.numpy as jnp
from typing import Tuple

def compute_stress_fa(stress_tensors: jnp.ndarray) -> jnp.ndarray:
    """
    Computes Fractional Anisotropy (FA) from Stress Tensors.
    
    FA = sqrt(3/2) * sqrt( sum((lambda_i - lambda_mean)^2) ) / sqrt( sum(lambda_i^2) )
    
    Args:
        stress_tensors: (..., 3, 3) arrays.
        
    Returns:
        fa_map: (...) map of FA values.
    """
    # 1. Eigenvalues
    # stress_tensors should be symmetric.
    # Force symmetry for stability? T_sym = 0.5 * (T + T.T)
    T_sym = 0.5 * (stress_tensors + jnp.swapaxes(stress_tensors, -1, -2))
    
    # eigh for symmetric matrices
    eigvals = jnp.linalg.eigvalsh(T_sym) # (..., 3)
    
    # 2. Mean Eigenvalue (Trace/3 / hydrostatic stress)
    mean_ev = jnp.mean(eigvals, axis=-1, keepdims=True)
    
    # 3. Numerator: Variance of eigenvalues
    # sum (lambda - mean)^2
    numerator = jnp.sqrt(jnp.sum((eigvals - mean_ev)**2, axis=-1))
    
    # 4. Denominator: Magnitude of eigenvalues
    # sum lambda^2
    denominator = jnp.sqrt(jnp.sum(eigvals**2, axis=-1))
    
    # Avoid division by zero
    denominator = jnp.where(denominator < 1e-9, 1e-9, denominator)
    
    # 5. FA
    fa = jnp.sqrt(1.5) * numerator / denominator
    
    # Clamp to [0, 1] just in case
    fa = jnp.clip(fa, 0.0, 1.0)
    
    return fa
