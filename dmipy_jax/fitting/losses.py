import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

def mse_loss(params, acquisition, data, model_func, unwrap_fn=None):
    """
    Computes Mean Squared Error loss.
    """
    if unwrap_fn:
        args = unwrap_fn(params)
        prediction = model_func(*args, acquisition)
    else:
        prediction = model_func(params, acquisition)
        
    return jnp.mean((prediction - data) ** 2)

def rician_nll_loss(params, acquisition, data, sigma, model_func, unwrap_fn=None):
    """
    Computes Negative Log-Likelihood for Rician distributed data.
    """
    if unwrap_fn:
        args = unwrap_fn(params)
        prediction = model_func(*args, acquisition)
    else:
        prediction = model_func(params, acquisition)

    # Numerical stability for Bessel function Io
    # log(Io(z)) approx z for large z
    # We use jsp.i0e(z) = Io(z) * exp(-|z|)
    # Io(z) = i0e(z) * exp(|z|)
    # log(Io(z)) = log(i0e(z)) + |z| (since z > 0 usually)
    
    z = (prediction * data) / (sigma ** 2)
    # z is non-negative since prediction and data are magnitudes
    
    i0e_val = jsp.i0e(z)
    # i0e_val can be very small but positive. 
    
    log_i0e = jnp.log(i0e_val + 1e-12) + z
    
    # NLL = -log(Signal / sigma^2) - log(Io(z)) + (Signal^2 + Prediction^2) / (2*sigma^2)
    # We ignore constant term -log(Signal/sigma^2) for optimization usually? 
    # Actually the full PDF is:
    # P(M|A, s) = (M/s^2) * exp(-(M^2+A^2)/(2s^2)) * Io(MA/s^2)
    # log P = log(M) - 2log(s) - (M^2+A^2)/(2s^2) + log(Io(z))
    # NLL = -log P ~ (M^2+A^2)/(2s^2) - log(Io(z))
    # s = sigma, M = data, A = prediction
    
    sse_term = (data ** 2 + prediction ** 2) / (2 * sigma ** 2)
    nll = sse_term - log_i0e
    
    return jnp.mean(nll)
