"""Jacobian-based uncertainty utilities for linearised error propagation.

Provides functions for computing the Jacobian of forward models with
respect to tissue parameters and propagating parameter covariance through
the linearised model (first-order Taylor expansion).  This gives a fast
approximation to output uncertainty when full Bayesian inference is too
expensive.

Key functions:

* :func:`compute_jacobian` -- computes the ``(n_measurements, n_params)``
  Jacobian matrix of a forward model at a given parameter vector using
  ``jax.jacfwd``.
* :func:`propagate_covariance` -- propagates a parameter covariance matrix
  through the Jacobian to obtain the output signal covariance:
  ``Sigma_y = J @ Sigma_theta @ J^T``.
* :func:`cramer_rao_lower_bound` -- computes the Cramer-Rao lower bound
  on parameter variance from the Fisher information matrix.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple
from jaxtyping import Array, Float

def compute_jacobian(
    model_func: Callable,
    params: Float[Array, " num_params"],
    acquisition: object
) -> Float[Array, " num_measurements num_params"]:
    """
    Computes the Jacobian of the model function with respect to parameters.
    
    Args:
        model_func: Function f(params, acquisition) -> signal
        params: Parameter vector at which to evaluate Jacobian.
        acquisition: Acquisition object passed to model_func.
        
    Returns:
        Jacobian matrix J of shape (N_measurements, N_params).
    """
    return jax.jacfwd(model_func)(params, acquisition)

def compute_fisher_information(
    jacobian: Float[Array, " num_measurements num_params"],
    sigma: float = 1.0
) -> Float[Array, " num_params num_params"]:
    """
    Computes the approximate Fisher Information Matrix (FIM) for Gaussian noise.
    FIM_ij = (1/sigma^2) * sum_k (dS_k/dtheta_i * dS_k/dtheta_j)
    In matrix form: FIM = (J.T @ J) / sigma^2
    
    Args:
        jacobian: Jacobian matrix J.
        sigma: Standard deviation of noise.
        
    Returns:
        Fisher Information Matrix.
    """
    return jnp.dot(jacobian.T, jacobian) / (sigma ** 2)

def compute_covariance(
    jacobian: Float[Array, " num_measurements num_params"],
    sigma: float = 1.0,
    epsilon: float = 1e-9
) -> Float[Array, " num_params num_params"]:
    """
    Computes the parameter covariance matrix as the inverse of the FIM.
    Cov = inv(FIM) = sigma^2 * inv(J.T @ J)
    
    Args:
        jacobian: Jacobian matrix J.
        sigma: Standard deviation of noise.
        epsilon: Small regularization term for numerical stability of inversion.
        
    Returns:
        Covariance matrix.
    """
    fim = compute_fisher_information(jacobian, sigma=1.0) # Scale sigma later
    
    # Regularize FIM before inversion to handle ill-conditioned cases
    fim_reg = fim + epsilon * jnp.eye(fim.shape[0])
    
    inv_fim = jnp.linalg.inv(fim_reg)
    
    return inv_fim * (sigma ** 2)

def compute_crlb_std(
    jacobian: Float[Array, " num_measurements num_params"],
    sigma: float = 1.0
) -> Float[Array, " num_params"]:
    """
    Computes the Cramer-Rao Lower Bound (Standard Deviation) for parameters.
    CRLB_i = sqrt(Cov_ii)
    
    Args:
        jacobian: Jacobian matrix J.
        sigma: Standard deviation of noise.
        
    Returns:
        Array of parameter standard deviations (uncertainties).
    """
    cov = compute_covariance(jacobian, sigma)
    # Ensure non-negative before sqrt (numerical noise might cause slightly negative diag)
    diag_cov = jnp.maximum(jnp.diag(cov), 0.0)
    return jnp.sqrt(diag_cov)
