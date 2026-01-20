<<<<<<< HEAD
import jax.numpy as jnp
import jax.scipy.special as jsp

__all__ = ['mse_loss', 'rician_nll_loss']

def mse_loss(params, acquisition, data, model_func, unwrap_fn=None):
    """
    Standard Mean Squared Error Loss.
    
    Args:
        params: Model parameters (dictionary or flat array depending on unwrap_fn).
        acquisition: JaxAcquisition object.
        data: Observed signal array.
        model_func: Function f(params, acquisition) -> predicted_signal.
        unwrap_fn: Optional function to process params before passing to model_func.
                   If None, params is passed directly.
    
    Returns:
        Scalar Mean Squared Error.
    """
    if unwrap_fn is not None:
        args = unwrap_fn(params)
        prediction = model_func(acquisition, *args) # Assuming unwrap returns args list? 
                                                    # Wait, standard composition in this repo might vary. 
                                                    # The JaxMultiCompartmentModel.model_func takes (params_flat, acq).
                                                    # While optimization.mse_loss takes (bvals, bvecs, *args).
                                                    # I should interpret this generically: prediction = model(params, acq)
    else:
        # If no unwrap, assume params matches model_func signature or model_func handles it
        prediction = model_func(params, acquisition)
        
    return jnp.mean((data - prediction) ** 2)


def rician_nll_loss(params, acquisition, data, sigma, model_func, unwrap_fn=None):
    """
    Negative Log-Likelihood for Rician Distributed Noise.
    
    Utilizes the approximation:
    NLL ~ -log(I0e(z)) + (S_prediction - S_data)^2 / (2 * sigma^2)
    where z = (S_prediction * S_data) / sigma^2
    I0e(z) = I_0(z) * exp(-z) is the exponentially scaled Bessel function of order 0.
    
    This form avoids overflow for large arguments of I_0, providing numerical stability.
    
    Args:
        params: Model parameters.
        acquisition: JaxAcquisition object.
        data: Observed signal array.
        sigma: Noise standard deviation (scalar or array).
        model_func: Function f(params, acquisition) -> predicted_signal.
        unwrap_fn: Optional function to process params.
        
    Returns:
        Scalar Negative Log-Likelihood (mean over batch).
    """
    if unwrap_fn is not None:
         args = unwrap_fn(params)
         prediction = model_func(acquisition, *args)
    else:
         prediction = model_func(params, acquisition)

    # Ensure precision
    # Use jnp.clip to avoid division by zero or log of zero if necessary, though logic handles it.
    
    # z calculation
    # z = S_pred * S_data / sigma^2
    z = (prediction * data) / (sigma ** 2)
    
    # i0e calculation
    i0e_val = jsp.i0e(z)
    
    # Log term: log(i0e(z))
    # Add epsilon to avoid log(0) if z is somehow such that i0e is 0 (unlikely for real inputs but good practice)
    log_i0e = jnp.log(i0e_val + 1e-12)

    # SSE term (scaled)
    # (S_pred - S_data)^2 / (2 * sigma^2)
    sse_term = ((prediction - data) ** 2) / (2 * sigma ** 2)
    
    nll = -log_i0e + sse_term
=======
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
>>>>>>> recovery_work_v2
    
    return jnp.mean(nll)
