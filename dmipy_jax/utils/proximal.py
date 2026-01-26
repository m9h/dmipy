import jax.numpy as jnp
from jax import jit

@jit
def soft_thresholding(x, lambda_val):
    """
    Applies the soft thresholding operator (proximal operator for L1 norm).
    
    prox_L1(x) = sign(x) * max(|x| - lambda, 0)
    
    Args:
        x: Input array.
        lambda_val: Threshold value (scalar or array broadcastable to x).
        
    Returns:
        Array with soft thresholding applied.
    """
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lambda_val, 0.0)

@jit
def non_negative_soft_thresholding(x, lambda_val):
    """
    Applies soft thresholding followed by projection to non-negative orthant.
    Useful for volume fractions.
    
    prox(x) = max(0, x - lambda)  (if x>0, else 0)
    Actually:
    prox = max(0, sign(x)*max(|x|-lambda, 0)) ??
    For x >= 0 constraint + L1:
    minimize 0.5(y-x)^2 + lambda*|x| s.t. x>=0
    x must be >= 0.
    The gradient of 0.5(y-x)^2 is -(y-x).
    Subgradient of lambda*x is lambda (since x>0).
    -(y-x) + lambda = 0 => x = y - lambda.
    Constraint x >= 0 => x = max(0, y - lambda).
    
    Args:
        x: Input array.
        lambda_val: Threshold.
        
    Returns:
        Array with non-negative soft thresholding.
    """
    return jnp.maximum(0.0, x - lambda_val)
