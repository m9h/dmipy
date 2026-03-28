"""DESPOT2 (SSFP/bSSFP T2 mapping) — JAX implementation with Laplace uncertainty.

Signal model (on-resonance balanced SSFP):
    S = M0 * sin(α) * (1 - E1) / (1 - (E1 - E2)*cos(α) - E1*E2)
    where E1 = exp(-TR/T1), E2 = exp(-TR/T2)

Requires T1 as known input (from DESPOT1/VFA fitting).
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


@jax.jit
def ssfp_signal(T1: float, T2: float, M0: float, TR: float,
                flip_angles: jnp.ndarray) -> jnp.ndarray:
    """Balanced SSFP signal (on-resonance, no banding)."""
    E1 = jnp.exp(-TR / T1)
    E2 = jnp.exp(-TR / T2)
    num = M0 * jnp.sin(flip_angles) * (1 - E1)
    den = 1 - (E1 - E2) * jnp.cos(flip_angles) - E1 * E2
    return num / den


class DESPOT2Result(NamedTuple):
    T2: float
    M0: float
    T2_std: float
    M0_std: float
    residual: float


def fit_despot2(
    data: jnp.ndarray,
    flip_angles: jnp.ndarray,
    TR: float,
    T1: float,
    n_iter: int = 30,
) -> DESPOT2Result:
    """Fit T2 and M0 from SSFP data given known T1."""
    E1 = jnp.exp(-TR / T1)

    # Grid search initialization over T2 candidates
    # SSFP is tricky — linear DESPOT2 linearization is poorly conditioned
    # Instead, find the T2 that minimizes loss over a coarse grid
    T2_candidates = jnp.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.3, 0.5, 1.0, 2.0])

    def grid_loss(T2_cand):
        E2_cand = jnp.exp(-TR / T2_cand)
        # Estimate M0 for this T2 via least-squares scale factor
        pred_unit = ssfp_signal(T1, T2_cand, 1.0, TR, flip_angles)
        M0_cand = jnp.sum(data * pred_unit) / jnp.maximum(jnp.sum(pred_unit ** 2), 1e-10)
        pred = pred_unit * M0_cand
        return jnp.mean((pred - data) ** 2)

    grid_losses = jax.vmap(grid_loss)(T2_candidates)
    best_idx = jnp.argmin(grid_losses)
    T2_init = T2_candidates[best_idx]

    # M0 from best T2
    pred_unit = ssfp_signal(T1, T2_init, 1.0, TR, flip_angles)
    M0_init = jnp.sum(data * pred_unit) / jnp.maximum(jnp.sum(pred_unit ** 2), 1e-10)
    M0_init = jnp.maximum(M0_init, 1.0)

    params = jnp.array([T2_init, M0_init])

    def loss_fn(params):
        t2 = jnp.maximum(params[0], 0.001)
        m0 = jnp.maximum(params[1], 0.001)
        pred = ssfp_signal(T1, t2, m0, TR, flip_angles)
        return jnp.mean((pred - data) ** 2)

    def gn_step(params, _):
        g = jax.grad(loss_fn)(params)
        H = jax.hessian(loss_fn)(params)
        damping = jnp.diag(jnp.abs(jnp.diag(H)) * 0.01 + 1e-6)
        step = jnp.linalg.solve(H + damping, g)
        new_params = params - step
        return jnp.maximum(new_params, jnp.array([0.001, 0.001])), None

    params, _ = jax.lax.scan(gn_step, params, None, length=n_iter)

    T2_fit = params[0]
    M0_fit = params[1]
    pred = ssfp_signal(T1, T2_fit, M0_fit, TR, flip_angles)
    residual = jnp.mean((pred - data) ** 2)

    N = data.shape[0]
    sigma_sq = jnp.maximum(jnp.sum((pred - data) ** 2) / jnp.maximum(N - 2, 1), 1e-6)
    H = jax.hessian(loss_fn)(params)
    cov = sigma_sq * jnp.linalg.inv(H + 1e-6 * jnp.eye(2))

    T2_std = jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))
    M0_std = jnp.sqrt(jnp.maximum(cov[1, 1], 0.0))

    return DESPOT2Result(T2=T2_fit, M0=M0_fit, T2_std=T2_std, M0_std=M0_std, residual=residual)


fit_despot2_batch = jax.vmap(fit_despot2, in_axes=(0, None, None, 0))
