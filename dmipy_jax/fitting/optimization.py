import jax
import jax.numpy as jnp
import jaxopt

__all__ = ['fit_voxel', 'ConstrainedOptimizer']

def fit_voxel(model_func, init_params, data, bounds=None):
    """
    Fit a single voxel signal using JAXopt's ScipyMinimize (L-BFGS-B).

    This function minimizes the Sum of Squared Errors (SSE) between the model prediction
    and the observed data. It supports bounded optimization if bounds are provided.

    Args:
        model_func (callable): A JAX-compatible function ``f(params) -> signal``.
            It must take a 1D array of parameters and return a 1D array of signal values
            matching the size of ``data``.
        init_params (jax.numpy.ndarray): Initial guess for the model parameters.
            Shape must match the input expected by ``model_func``.
        data (jax.numpy.ndarray): Measured signal intensity for the voxel.
            Shape: (N_measurements,).
        bounds (tuple of jax.numpy.ndarray, optional): A tuple ``(min_bounds, max_bounds)``
            defining lower and upper constraints for each parameter. Each array must be
            broadcastable to ``init_params``. If None, unconstrained optimization is performed.

    Returns:
        tuple:
            - **params** (*jax.numpy.ndarray*): The optimized model parameters.
            - **state** (*jaxopt.ScipyMinimizeInfo*): The final state of the optimizer,
              containing convergence information.
    """
    
    # Define the objective function: Sum of Squared Errors
    def objective(params):
        # We also apply bounds inside if needed for safety, but rely on optimizer.
        p = params
        if bounds is not None:
             lower, upper = bounds
             p = jnp.clip(p, lower, upper)
        
        predictions = model_func(p)
        residuals = predictions - data
        return jnp.sum(residuals**2)

    if bounds is not None:
        # Use ScipyBoundedMinimize for constrained validation
        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method='L-BFGS-B')
        res = solver.run(init_params, bounds=bounds)
    else:
        # Use ScipyMinimize for unconstrained
        solver = jaxopt.ScipyMinimize(fun=objective, method='L-BFGS-B')
        res = solver.run(init_params)
        
    return res.params, res.state


class ConstrainedOptimizer:
    """
    A generic optimizer wrapper that supports Penalized Likelihood (Priors).
    
    It minimizes: Loss(data, model(params)) - Sum(log_priors(params))
    
    Args:
        model_func (callable): function f(params) -> predictions
        priors (list of callable, optional): List of functions doing params -> log_prob.
            The return values are summed and SUBTRACTED from the loss.
        loss_metric (callable, optional): function f(predictions, data) -> scalar loss.
            Default is Sum of Squared Errors (SSE).
    """
    def __init__(self, model_func, priors=None, loss_metric=None):
        self.model_func = model_func
        self.priors = priors if priors is not None else []
        if loss_metric is None:
            self.loss_metric = lambda preds, data: jnp.sum((preds - data)**2)
        else:
            self.loss_metric = loss_metric

    def fit(self, init_params, data, bounds=None):
        """
        Run the optimization.
        """
        def objective(params):
            p = params
            if bounds is not None:
                lower, upper = bounds
                p = jnp.clip(p, lower, upper)
            
            preds = self.model_func(p)
            current_loss = self.loss_metric(preds, data)
            
            # Prior penalty
            # maximize log_likelihood + log_prior => minimize -log_likelihood - log_prior
            # If loss is SSE (~ -log_likelihood), then minimize SSE - log_prior
            
            prior_term = 0.0
            for prior_func in self.priors:
                prior_term += prior_func(p)
                
            return current_loss - prior_term

        if bounds is not None:
            solver = jaxopt.ScipyBoundedMinimize(fun=objective, method='L-BFGS-B')
            res = solver.run(init_params, bounds=bounds)
        else:
            solver = jaxopt.ScipyMinimize(fun=objective, method='L-BFGS-B')
            res = solver.run(init_params)
            
        return res.params, res.state
