
import jax.numpy as jnp

def inverse_softplus(y):
    # y = log(1 + exp(x))
    # exp(y) = 1 + exp(x)
    # exp(x) = exp(y) - 1
    # x = log(exp(y) - 1)
    
    # Numerical stability for small y (where softplus(x) ~ exp(x) -> x ~ log(y))
    # And large y (where softplus(x) ~ x)
    
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))
