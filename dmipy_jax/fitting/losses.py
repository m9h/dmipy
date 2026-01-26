
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

__all__ = ['mse_loss', 'rician_nll_loss', 'prior_loss']

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


def prior_loss(params, acquisition, data, coords, material_map, 
               param_indices={'mu': 0, 'alpha': 1}, 
               weights={'mu': 1.0, 'alpha': 1.0},
               unwrap_fn=None):
    """
    Bayesian Prior Loss based on Kuhl Lab brain tissue maps.
    
    L_prior = || theta_predicted - theta_Kuhl ||_weighted^2
    
    Args:
        params: Model parameters.
        acquisition: JaxAcquisition object (unused but kept for signature consistency).
        data: Observed data (unused but kept for signature consistency).
        coords: (N, 3) MNI coordinates corresponding to the data/params.
        material_map: BrainMaterialMap instance.
        param_indices: Dict mapping 'mu' and 'alpha' to indices in the parameter vector.
                       Assumes params is (N, P).
        weights: Dict with 'mu' and 'alpha' weights.
        unwrap_fn: Optional unwrap.
        
    Returns:
        Scalar loss.
    """
    if unwrap_fn:
        # If unwrapped, we assume params is a list/tuple of arrays
        # This makes it hard to index by "param_indices" directly unless we know the structure.
        # For this specific loss, let's assume 'params' is the packed array if unwrap_fn is None,
        # OR unwrap_fn returns a dictionary or struct we can access?
        # To keep it compatible with typical usage, let's assume params is the raw fitted parameters.
        # If unwrap_fn is provided, we probably apply it to get physical parameters.
        # Let's assume unwrap_fn(params) returns a dict or something indexable?
        # Actually, standard dmipy-jax pattern (as seen in mse_loss):
        # args = unwrap_fn(params) -> prediction = model_func(*args)
        # Here we don't have model_func. We inspect params directly.
        pass
        
    # Get priors
    # priors: (N, 2) -> [mu_target, alpha_target]
    priors = material_map.get_priors(coords)
    mu_target = priors[..., 0]
    alpha_target = priors[..., 1]
    
    # Get predictions
    # Assuming params is (N, P) or similar
    # We need to know which param is mu and which is alpha.
    mu_idx = param_indices['mu']
    alpha_idx = param_indices['alpha']
    
    # Handle case where params might be list of arrays (common in some JAX optimizers)
    if isinstance(params, (list, tuple)):
         mu_pred = params[mu_idx]
         alpha_pred = params[alpha_idx]
    else:
         mu_pred = params[..., mu_idx]
         alpha_pred = params[..., alpha_idx]
         
    loss_mu = weights['mu'] * jnp.mean((mu_pred - mu_target)**2)
    loss_alpha = weights['alpha'] * jnp.mean((alpha_pred - alpha_target)**2)
    
    return loss_mu + loss_alpha
