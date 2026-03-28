"""mcDESPOT — Multi-Component DESPOT (Myelin Water Fraction).

Fits a two-pool model to combined SPGR + bSSFP data to decompose T1/T2
into fast (myelin water) and slow (intra/extra-cellular water) pools.

Parameters (6):
  MWF:  myelin water fraction [0, 1]
  T1_f: fast pool T1 (seconds) — typically ~0.3-0.5s
  T1_s: slow pool T1 (seconds) — typically ~1.0-1.5s
  T2_f: fast pool T2 (seconds) — typically ~10-20ms
  T2_s: slow pool T2 (seconds) — typically ~60-100ms
  M0:   total signal amplitude

Signal: weighted sum of SPGR and SSFP signals from each pool.

Reference: Deoni SCL et al. (2008). Gleaning multicomponent T1 and T2
information from steady-state imaging data. MRM 60(6):1372-1387.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp

from dmipy_jax.signal_models.qmri.vfa_t1 import spgr_signal
from dmipy_jax.signal_models.qmri.despot2 import ssfp_signal


class McDESPOTResult(NamedTuple):
    MWF: float
    T1_fast: float
    T1_slow: float
    T2_fast: float
    T2_slow: float
    M0: float
    MWF_std: float
    residual: float


@jax.jit
def mcdespot_spgr_signal(
    MWF: float, T1_f: float, T1_s: float, M0: float,
    TR: float, flip_angles: jnp.ndarray
) -> jnp.ndarray:
    """Two-pool SPGR signal (sum of fast and slow T1 pools)."""
    S_fast = spgr_signal(T1=T1_f, M0=M0 * MWF, TR=TR, flip_angles=flip_angles)
    S_slow = spgr_signal(T1=T1_s, M0=M0 * (1 - MWF), TR=TR, flip_angles=flip_angles)
    return S_fast + S_slow


@jax.jit
def mcdespot_ssfp_signal(
    MWF: float, T1_f: float, T1_s: float, T2_f: float, T2_s: float, M0: float,
    TR: float, flip_angles: jnp.ndarray
) -> jnp.ndarray:
    """Two-pool SSFP signal (sum of fast and slow T1/T2 pools)."""
    S_fast = ssfp_signal(T1=T1_f, T2=T2_f, M0=M0 * MWF, TR=TR, flip_angles=flip_angles)
    S_slow = ssfp_signal(T1=T1_s, T2=T2_s, M0=M0 * (1 - MWF), TR=TR, flip_angles=flip_angles)
    return S_fast + S_slow


@jax.jit
def mcdespot_combined_signal(
    params: jnp.ndarray,
    spgr_fa: jnp.ndarray, spgr_TR: float,
    ssfp_fa: jnp.ndarray, ssfp_TR: float,
) -> jnp.ndarray:
    """Combined SPGR + SSFP signal for fitting.

    params: [MWF, T1_f, T1_s, T2_f, T2_s, M0]
    Returns: concatenated [SPGR_signals, SSFP_signals]
    """
    MWF = jnp.clip(params[0], 0.01, 0.99)
    T1_f = jnp.maximum(params[1], 0.01)
    T1_s = jnp.maximum(params[2], 0.01)
    T2_f = jnp.maximum(params[3], 0.001)
    T2_s = jnp.maximum(params[4], 0.001)
    M0 = jnp.maximum(params[5], 0.001)

    spgr = mcdespot_spgr_signal(MWF, T1_f, T1_s, M0, spgr_TR, spgr_fa)
    ssfp = mcdespot_ssfp_signal(MWF, T1_f, T1_s, T2_f, T2_s, M0, ssfp_TR, ssfp_fa)
    return jnp.concatenate([spgr, ssfp])


def fit_mcdespot(
    spgr_data: jnp.ndarray,
    ssfp_data: jnp.ndarray,
    spgr_fa: jnp.ndarray,
    ssfp_fa: jnp.ndarray,
    spgr_TR: float,
    ssfp_TR: float,
    n_iter: int = 50,
) -> McDESPOTResult:
    """Fit mcDESPOT two-pool model with Laplace uncertainty.

    Grid search initialization → Gauss-Newton refinement.
    """
    combined_data = jnp.concatenate([spgr_data, ssfp_data])

    def loss_fn(params):
        pred = mcdespot_combined_signal(params, spgr_fa, spgr_TR, ssfp_fa, ssfp_TR)
        return jnp.mean((pred - combined_data) ** 2)

    # Grid search over MWF and T2_fast (most sensitive parameters)
    # Fix T1_fast, T1_slow, T2_slow, M0 to physiological defaults
    mwf_grid = jnp.array([0.05, 0.1, 0.15, 0.2, 0.3])
    t2f_grid = jnp.array([0.010, 0.015, 0.020, 0.030])

    # Estimate M0 and T1s from SPGR data (single-pool VFA as starting point)
    from dmipy_jax.signal_models.qmri.vfa_t1 import fit_vfa_t1
    vfa_result = fit_vfa_t1(spgr_data, spgr_fa, spgr_TR)
    T1_init = vfa_result.T1
    M0_init = vfa_result.M0

    best_loss = jnp.inf
    best_params = jnp.array([0.15, 0.4, T1_init, 0.015, 0.07, M0_init])

    def eval_grid(mwf, t2f):
        params = jnp.array([mwf, 0.4, jnp.maximum(T1_init, 0.5), t2f, 0.07, jnp.maximum(M0_init, 1.0)])
        return loss_fn(params)

    # Vectorized grid search
    losses = jax.vmap(jax.vmap(eval_grid, in_axes=(None, 0)), in_axes=(0, None))(mwf_grid, t2f_grid)
    best_idx = jnp.unravel_index(jnp.argmin(losses), losses.shape)
    best_mwf = mwf_grid[best_idx[0]]
    best_t2f = t2f_grid[best_idx[1]]

    params = jnp.array([best_mwf, 0.4, jnp.maximum(T1_init, 0.5),
                         best_t2f, 0.07, jnp.maximum(M0_init, 1.0)])

    # Gauss-Newton refinement
    def gn_step(params, _):
        g = jax.grad(loss_fn)(params)
        H = jax.hessian(loss_fn)(params)
        damping = jnp.diag(jnp.abs(jnp.diag(H)) * 0.05 + 1e-6)
        step = jnp.linalg.solve(H + damping, g)
        new_params = params - step * 0.5  # conservative step
        # Enforce bounds
        new_params = new_params.at[0].set(jnp.clip(new_params[0], 0.01, 0.5))   # MWF
        new_params = new_params.at[1].set(jnp.clip(new_params[1], 0.1, 1.0))    # T1_fast
        new_params = new_params.at[2].set(jnp.clip(new_params[2], 0.3, 3.0))    # T1_slow
        new_params = new_params.at[3].set(jnp.clip(new_params[3], 0.005, 0.05)) # T2_fast
        new_params = new_params.at[4].set(jnp.clip(new_params[4], 0.03, 0.3))   # T2_slow
        new_params = new_params.at[5].set(jnp.maximum(new_params[5], 0.001))     # M0
        return new_params, None

    params, _ = jax.lax.scan(gn_step, params, None, length=n_iter)

    pred = mcdespot_combined_signal(params, spgr_fa, spgr_TR, ssfp_fa, ssfp_TR)
    residual = jnp.mean((pred - combined_data) ** 2)

    # Uncertainty (Laplace on MWF primarily)
    N = combined_data.shape[0]
    sigma_sq = jnp.maximum(jnp.sum((pred - combined_data) ** 2) / jnp.maximum(N - 6, 1), 1e-6)
    H = jax.hessian(loss_fn)(params)
    cov = sigma_sq * jnp.linalg.inv(H + 1e-6 * jnp.eye(6))
    MWF_std = jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))

    return McDESPOTResult(
        MWF=params[0], T1_fast=params[1], T1_slow=params[2],
        T2_fast=params[3], T2_slow=params[4], M0=params[5],
        MWF_std=MWF_std, residual=residual,
    )
