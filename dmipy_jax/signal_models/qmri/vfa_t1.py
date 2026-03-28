"""Variable Flip Angle T1 mapping (DESPOT1) — JAX implementation.

Fits T1 and M0 (proton density) from SPGR/FLASH data acquired at
multiple flip angles, with Laplace-approximation uncertainty.

Signal model (Ernst equation):
    S(α) = M0 · sin(α) · (1 - E1) / (1 - E1 · cos(α))
    where E1 = exp(-TR/T1)

Fitting: nonlinear least squares via JAX autodiff + Newton's method,
with Hessian-based uncertainty (Laplace approximation, like FABBER).

References
----------
Deoni SCL, Peters TM, Rutt BK (2003). High-resolution T1 and T2
mapping of the brain in a clinically acceptable time with DESPOT1
and DESPOT2. MRM 49(3):515-526.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------

@jax.jit
def spgr_signal(
    T1: float,
    M0: float,
    TR: float,
    flip_angles: jnp.ndarray,
) -> jnp.ndarray:
    """SPGR/FLASH steady-state signal (Ernst equation).

    Parameters
    ----------
    T1 : float — longitudinal relaxation time (seconds).
    M0 : float — equilibrium magnetization (proton density).
    TR : float — repetition time (seconds).
    flip_angles : (N,) — flip angles in radians.

    Returns
    -------
    signal : (N,) — signal intensity at each flip angle.
    """
    E1 = jnp.exp(-TR / T1)
    return M0 * jnp.sin(flip_angles) * (1 - E1) / (1 - E1 * jnp.cos(flip_angles))


# ---------------------------------------------------------------------------
# Fit result with uncertainty
# ---------------------------------------------------------------------------

class VFAResult(NamedTuple):
    """Result of VFA T1 fitting with uncertainty.

    Attributes
    ----------
    T1 : float — fitted T1 (seconds).
    M0 : float — fitted M0 (proton density).
    T1_std : float — standard deviation of T1 estimate (from Hessian).
    M0_std : float — standard deviation of M0 estimate.
    residual : float — mean squared residual.
    """
    T1: float
    M0: float
    T1_std: float
    M0_std: float
    residual: float


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_vfa_t1(
    data: jnp.ndarray,
    flip_angles: jnp.ndarray,
    TR: float,
    b1_scale: float = 1.0,
    n_iter: int = 50,
    lr: float = 0.01,
) -> VFAResult:
    """Fit T1 and M0 from VFA SPGR data with uncertainty.

    Uses gradient descent in log-space (ensures T1, M0 > 0) followed
    by Hessian-based uncertainty estimation (Laplace approximation).

    Parameters
    ----------
    data : (N,) — measured signal at each flip angle.
    flip_angles : (N,) — nominal flip angles in radians.
    TR : float — repetition time (seconds).
    b1_scale : float — B1+ scaling factor (1.0 = no correction).
        Actual flip angle = nominal * b1_scale.
    n_iter : int — number of gradient descent iterations.
    lr : float — learning rate.

    Returns
    -------
    VFAResult with T1, M0, T1_std, M0_std, residual.
    """
    # Apply B1 correction
    actual_fa = flip_angles * b1_scale

    # Initialize via linear DESPOT1 (Deoni 2003)
    # Linearize: S/sin(α) = E1 · S/tan(α) + M0·(1-E1)
    y = data / jnp.sin(actual_fa)
    x = data / jnp.tan(actual_fa)

    # Linear regression: y = slope * x + intercept
    x_mean = jnp.mean(x)
    y_mean = jnp.mean(y)
    slope = jnp.sum((x - x_mean) * (y - y_mean)) / jnp.maximum(
        jnp.sum((x - x_mean) ** 2), 1e-20
    )
    intercept = y_mean - slope * x_mean

    # E1 = slope, M0 = intercept / (1 - E1)
    E1_init = jnp.clip(slope, 0.01, 0.9999)
    T1_init = -TR / jnp.log(E1_init)
    M0_init = intercept / jnp.maximum(1 - E1_init, 1e-10)

    # Use linear DESPOT1 solution as starting point, then refine
    # with Gauss-Newton steps in natural (not log) space
    T1_fit = jnp.maximum(T1_init, 0.01)
    M0_fit = jnp.maximum(M0_init, 1.0)

    # Loss in natural parameter space
    def loss_fn(params):
        T1 = jnp.maximum(params[0], 0.001)
        M0 = jnp.maximum(params[1], 0.001)
        predicted = spgr_signal(T1, M0, TR, actual_fa)
        return jnp.mean((predicted - data) ** 2)

    params = jnp.array([T1_fit, M0_fit])

    # Gauss-Newton refinement with Levenberg-Marquardt damping
    # Written to be vmap/jit compatible (no Python control flow on traced values)
    params = jnp.array([T1_fit, M0_fit])

    def gn_step(params, _):
        g = jax.grad(loss_fn)(params)
        H = jax.hessian(loss_fn)(params)
        damping = jnp.diag(jnp.abs(jnp.diag(H)) * 0.01 + 1e-6)
        step = jnp.linalg.solve(H + damping, g)
        new_params = params - step
        new_params = jnp.maximum(new_params, jnp.array([0.001, 0.001]))
        return new_params, None

    params, _ = jax.lax.scan(gn_step, params, None, length=n_iter)

    T1_fit = params[0]
    M0_fit = params[1]

    # Residual
    predicted = spgr_signal(T1_fit, M0_fit, TR, actual_fa)
    residual = jnp.mean((predicted - data) ** 2)

    # Uncertainty via Laplace approximation (Hessian-based, like FABBER)
    # Cov ≈ σ² · H^{-1}
    # Use data-driven noise estimate: σ² = RSS / max(N-2, 1)
    # For noiseless data, use Cramér-Rao bound instead: Cov = (J^T J)^{-1}
    N = data.shape[0]
    rss = jnp.sum((predicted - data) ** 2)
    sigma_sq = rss / jnp.maximum(N - 2, 1)
    # Floor sigma_sq to give meaningful uncertainty even for perfect fits
    sigma_sq = jnp.maximum(sigma_sq, 1e-6)

    H = jax.hessian(loss_fn)(params)
    H_reg = H + 1e-6 * jnp.eye(2)
    cov = sigma_sq * jnp.linalg.inv(H_reg)

    T1_std = jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))
    M0_std = jnp.sqrt(jnp.maximum(cov[1, 1], 0.0))

    return VFAResult(
        T1=T1_fit,
        M0=M0_fit,
        T1_std=T1_std,
        M0_std=M0_std,
        residual=residual,
    )


# Convenience: vmap-compatible version
fit_vfa_t1_batch = jax.vmap(fit_vfa_t1, in_axes=(0, None, None))
