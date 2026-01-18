
import jax
import jax.numpy as jnp
import pytest
from dmipy_jax.core.uncertainty_utils import compute_jacobian, compute_crlb_std
import numpy as np

def test_jacobian_linear_model():
    # Model: f(p, x) = p0 * x + p1
    def model(params, x):
        return params[0] * x + params[1]
    
    x = jnp.array([1.0, 2.0, 3.0])
    params = jnp.array([2.0, 1.0]) # y = 2x + 1 -> [3, 5, 7]
    
    # Analytical Jacobian:
    # df/dp0 = x
    # df/dp1 = 1
    # J = [[1, 1], [2, 1], [3, 1]]
    
    J = compute_jacobian(model, params, x)
    
    expected_J = jnp.stack([x, jnp.ones_like(x)], axis=1)
    
    assert jnp.allclose(J, expected_J)

def test_crlb_vs_scipy_curve_fit():
    # Verify that our CRLB matches standard statistical results for linear regression
    # Variance of estimator beta_hat = sigma^2 * (X.T X)^-1
    
    def model(params, x):
        return params[0] * x 
    
    x = jnp.linspace(0, 10, 20)
    true_params = jnp.array([2.5])
    
    # J for linear model p0*x is just x (column vector)
    J = compute_jacobian(model, true_params, x) # shape (20, 1)
    
    # FIM = J.T @ J = sum(x^2)
    # Var(p0) = sigma^2 / sum(x^2)
    
    sigma = 0.5
    
    crlb_std = compute_crlb_std(J, sigma=sigma)
    
    expected_var = (sigma**2) / jnp.sum(x**2)
    expected_std = jnp.sqrt(expected_var)
    
    assert jnp.allclose(crlb_std[0], expected_std, rtol=1e-5)

def test_multi_compartment_integration():
    # Mocking a MultiCompartmentModel scenario
    # We won't use the full class here to avoid heavy imports, but verify the logic used in fit()
    
    # create dummy data and model
    def simple_model(params, acq):
        # params: [A, B]
        # acq: just x values
        return params[0] * jnp.exp(-params[1] * acq)
    
    acq = jnp.array([0.1, 0.5, 1.0])
    true_params = jnp.array([1.0, 0.5])
    data = simple_model(true_params, acq)
    
    # Noisy data
    # (Not fitting here, just checking CRLB calculation pipeline)
    
    sigma_est = 0.1
    J = compute_jacobian(simple_model, true_params, acq)
    stds = compute_crlb_std(J, sigma=sigma_est)
    
    assert stds.shape == (2,)
    assert jnp.all(stds > 0)
