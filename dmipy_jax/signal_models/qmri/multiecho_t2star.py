"""Multi-echo T2* mapping — JAX implementation with Laplace uncertainty.

Signal model: S(TE) = S0 * exp(-TE / T2*)

Fits S0 and T2* from multi-echo gradient echo (MEGRE) data.
WAND ses-06 has 7 echoes for this purpose.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


@jax.jit
def multiecho_signal(S0: float, T2star: float, echo_times: jnp.ndarray) -> jnp.ndarray:
    """Mono-exponential T2* decay signal."""
    return S0 * jnp.exp(-echo_times / T2star)


class MultiEchoResult(NamedTuple):
    S0: float
    T2star: float
    S0_std: float
    T2star_std: float
    residual: float


def fit_multiecho_t2star(
    data: jnp.ndarray,
    echo_times: jnp.ndarray,
    n_iter: int = 20,
) -> MultiEchoResult:
    """Fit T2* from multi-echo data with Laplace uncertainty.

    Log-linear initialization → Gauss-Newton refinement → Hessian uncertainty.
    """
    # Log-linear init: log(S) = log(S0) - TE/T2*
    log_data = jnp.log(jnp.maximum(data, 1e-10))
    n = echo_times.shape[0]
    te_mean = jnp.mean(echo_times)
    log_mean = jnp.mean(log_data)
    slope = jnp.sum((echo_times - te_mean) * (log_data - log_mean)) / jnp.maximum(
        jnp.sum((echo_times - te_mean) ** 2), 1e-20
    )
    intercept = log_mean - slope * te_mean

    T2star_init = jnp.clip(-1.0 / jnp.minimum(slope, -1e-6), 0.001, 1.0)
    S0_init = jnp.exp(intercept)

    params = jnp.array([S0_init, T2star_init])

    def loss_fn(params):
        s0 = jnp.maximum(params[0], 0.001)
        t2s = jnp.maximum(params[1], 0.0001)
        pred = multiecho_signal(s0, t2s, echo_times)
        return jnp.mean((pred - data) ** 2)

    def gn_step(params, _):
        g = jax.grad(loss_fn)(params)
        H = jax.hessian(loss_fn)(params)
        damping = jnp.diag(jnp.abs(jnp.diag(H)) * 0.01 + 1e-6)
        step = jnp.linalg.solve(H + damping, g)
        new_params = params - step
        return jnp.maximum(new_params, jnp.array([0.001, 0.0001])), None

    params, _ = jax.lax.scan(gn_step, params, None, length=n_iter)

    S0_fit = params[0]
    T2star_fit = params[1]
    pred = multiecho_signal(S0_fit, T2star_fit, echo_times)
    residual = jnp.mean((pred - data) ** 2)

    # Laplace uncertainty
    N = data.shape[0]
    sigma_sq = jnp.maximum(jnp.sum((pred - data) ** 2) / jnp.maximum(N - 2, 1), 1e-6)
    H = jax.hessian(loss_fn)(params)
    cov = sigma_sq * jnp.linalg.inv(H + 1e-6 * jnp.eye(2))

    S0_std = jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))
    T2star_std = jnp.sqrt(jnp.maximum(cov[1, 1], 0.0))

    return MultiEchoResult(S0=S0_fit, T2star=T2star_fit,
                           S0_std=S0_std, T2star_std=T2star_std, residual=residual)


fit_multiecho_t2star_batch = jax.vmap(fit_multiecho_t2star, in_axes=(0, None))
