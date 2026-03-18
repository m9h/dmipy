"""
Phase 6: Acquisition-optimal SBI — expected information gain and
gradient-based acquisition optimisation for SBI training.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Any, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dmipy_jax.design.oed import compute_fisher_information, get_sensitivity


# --------------------------------------------------------------------------- #
# Expected Information Gain (EIG) via MC
# --------------------------------------------------------------------------- #

def expected_information_gain(
    flow,
    simulator: Callable,
    prior_sampler: Callable,
    acquisition: Any,
    key: PRNGKeyArray,
    n_outer: int = 256,
    n_inner: int = 128,
) -> float:
    """Monte-Carlo estimate of the Expected Information Gain.

    ``EIG = E_{theta, x}[ log q(theta | x) - log p(theta) ]``

    where ``q`` is the amortised posterior (flow/MDN) and ``p`` is the prior.

    Parameters
    ----------
    flow : eqx.Module
        Trained normalising flow with ``.log_prob(theta, condition=x)``.
    simulator : callable
        ``(key, theta_batch) -> signals``
    prior_sampler : callable
        ``(key, n) -> theta``
    acquisition : object
        Passed to simulator (but unused by it if already baked in).
    key : jax.Array
    n_outer, n_inner : int
        MC sample counts.

    Returns
    -------
    eig : float
        Estimated mutual information in nats.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Outer samples: theta ~ prior
    theta = prior_sampler(k1, n_outer)  # (n_outer, D)
    x = simulator(k2, theta)            # (n_outer, M)

    # log q(theta | x)
    log_q = jax.vmap(lambda t, s: flow.log_prob(t, condition=s))(theta, x)

    # log p(theta) ≈ -D * log(range)  (uniform prior → constant)
    # We compute a relative EIG; the prior term cancels in differences.
    # But for absolute EIG we approximate via marginal:
    # log p(theta) ≈ log (1/N_inner) Σ q(theta | x_j)  using inner samples

    theta_inner = prior_sampler(k3, n_inner)
    x_inner = simulator(k3, theta_inner)

    # For each outer theta, estimate log p(theta) ≈ logsumexp(log q(theta|x_j)) - log(n_inner)
    def marginal_log_prob(t):
        log_probs = jax.vmap(lambda s: flow.log_prob(t, condition=s))(x_inner)
        return jax.nn.logsumexp(log_probs) - jnp.log(n_inner)

    log_p_approx = jax.vmap(marginal_log_prob)(theta)

    eig = jnp.mean(log_q - log_p_approx)
    return eig


# --------------------------------------------------------------------------- #
# Fisher-based minimum measurements
# --------------------------------------------------------------------------- #

def fisher_minimum_measurements(
    model_func: Callable,
    params: Dict[str, Any],
    sigma: float = 1.0,
    acq_params: Optional[Dict[str, Any]] = None,
) -> int:
    """Estimate the minimum number of measurements for parameter identifiability.

    The Fisher Information Matrix must be full-rank (positive definite) for all
    parameters to be identifiable.  This function computes the FIM at the given
    operating point and returns its rank.

    Parameters
    ----------
    model_func : callable
    params : dict of tissue parameters
    sigma : float
    acq_params : dict  (must contain ``bvalues``, ``gradient_directions``)

    Returns
    -------
    rank : int
        Effective rank of FIM; if ``rank == n_params``, all parameters are
        identifiable with this acquisition.
    """
    if acq_params is None:
        raise ValueError("acq_params required")

    J = get_sensitivity(model_func, params, acq_params)
    fim = compute_fisher_information(J, sigma)
    # Numerical rank via SVD
    s = jnp.linalg.svd(fim, compute_uv=False)
    rank = int(jnp.sum(s > 1e-6 * s[0]))
    return rank


# --------------------------------------------------------------------------- #
# A-optimality loss
# --------------------------------------------------------------------------- #

def a_optimality_loss(
    trainable_acq_params: Dict[str, jnp.ndarray],
    static_acq_params: Dict[str, Any],
    model_func: Callable,
    target_params: Dict[str, Any],
    sigma: float = 1.0,
) -> float:
    """A-optimality criterion: trace of the inverse FIM.

    Minimising this is equivalent to minimising the total variance of the
    parameter estimates.

    Parameters
    ----------
    trainable_acq_params : dict
    static_acq_params : dict
    model_func : callable
    target_params : dict
    sigma : float

    Returns
    -------
    loss : float
    """
    acq_params = {**trainable_acq_params, **static_acq_params}

    # Normalise gradient directions
    vecs = acq_params["gradient_directions"]
    norms = jnp.linalg.norm(vecs, axis=1, keepdims=True)
    norms = jnp.where(norms == 0, 1.0, norms)
    acq_params["gradient_directions"] = vecs / norms

    J = get_sensitivity(model_func, target_params, acq_params)
    fim = compute_fisher_information(J, sigma)

    # Add jitter for invertibility
    n = fim.shape[0]
    fim_reg = fim + jnp.eye(n) * 1e-6

    fim_inv = jnp.linalg.inv(fim_reg)
    return jnp.trace(fim_inv)


# --------------------------------------------------------------------------- #
# Gradient-based acquisition optimisation for SBI
# --------------------------------------------------------------------------- #

def optimize_acquisition_for_sbi(
    initial_acq: Any,
    model_func: Callable,
    target_params: Dict[str, Any],
    n_steps: int = 200,
    learning_rate: float = 0.05,
    criterion: str = "d",
    sigma: float = 1.0,
    b_scale: float = 1e9,
    b_min: float = 0.0,
    b_max: float = 10_000e6,
) -> Tuple[Any, list]:
    """Optimise an acquisition scheme for SBI training via Fisher criteria.

    Parameters
    ----------
    initial_acq : JaxAcquisition
    model_func : callable
    target_params : dict
    n_steps : int
    learning_rate : float
    criterion : str  ``"d"`` (D-optimal) or ``"a"`` (A-optimal)
    sigma, b_scale, b_min, b_max : float

    Returns
    -------
    optimized_acq : JaxAcquisition
    loss_history : list of float
    """
    import copy
    from dmipy_jax.design.oed import d_optimality_loss

    loss_fn = d_optimality_loss if criterion == "d" else a_optimality_loss

    trainable = {
        "bvalues": (initial_acq.bvalues / b_scale).astype(jnp.float32),
        "gradient_directions": initial_acq.gradient_directions.astype(jnp.float32),
    }
    static = {}
    if initial_acq.delta is not None:
        static["delta"] = initial_acq.delta
    if initial_acq.Delta is not None:
        static["Delta"] = initial_acq.Delta

    def wrapper(t, s, m, p, sig):
        t_unscaled = {
            "bvalues": t["bvalues"] * b_scale,
            "gradient_directions": t["gradient_directions"],
        }
        return loss_fn(t_unscaled, s, m, p, sig)

    val_and_grad = jax.value_and_grad(wrapper, argnums=0)
    history = []

    for _ in range(n_steps):
        loss_val, grads = val_and_grad(trainable, static, model_func, target_params, sigma)
        history.append(float(loss_val))

        trainable["bvalues"] = trainable["bvalues"] - learning_rate * grads["bvalues"]
        trainable["gradient_directions"] = trainable["gradient_directions"] - learning_rate * grads["gradient_directions"]

        # Constraints
        trainable["bvalues"] = jnp.clip(trainable["bvalues"], b_min / b_scale, b_max / b_scale)
        vecs = trainable["gradient_directions"]
        norms = jnp.linalg.norm(vecs, axis=1, keepdims=True)
        norms = jnp.where(norms == 0, 1.0, norms)
        trainable["gradient_directions"] = vecs / norms

    new_acq = copy.copy(initial_acq)
    new_acq.bvalues = trainable["bvalues"] * b_scale
    new_acq.gradient_directions = trainable["gradient_directions"]
    return new_acq, history
