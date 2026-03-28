"""Quantitative Magnetization Transfer (qMT) — Ramani two-pool model.

Fits bound pool fraction (BPF/F) and forward exchange rate (kf) from
MT-weighted SPGR data at multiple offset frequencies and flip angles.

Two-pool model (Ramani et al. 2002, Sled & Pike 2001):
  Free pool (water): R1f, T2f, M0f
  Bound pool (macromolecular/myelin): F (fraction), kf (exchange), T2b

The observed signal depends on the saturation of the bound pool by
the MT pulse, which depends on offset frequency and power.

Simplified Ramani model for SPGR-qMT:
  Signal = M0 * (1 - F) * (R1_obs / R1f) * SPGR_signal
  where R1_obs depends on F, kf, and MT saturation

Reference: Ramani A et al. (2002). Precise estimate of fundamental
in-vivo MT parameters in human brain in clinically feasible times.
MRM 47(3):515-526.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


class QMTResult(NamedTuple):
    F: float        # bound pool fraction [0, 1]
    kf: float       # forward exchange rate (Hz)
    T2b: float      # bound pool T2 (seconds)
    F_std: float    # uncertainty on F
    kf_std: float   # uncertainty on kf
    residual: float


@jax.jit
def qmt_signal_ratio(
    F: float,
    kf: float,
    T2b: float,
    R1f: float,
    offset_freq: jnp.ndarray,
    mt_flip_angle: jnp.ndarray,
    mt_pulse_duration: float = 0.01,
) -> jnp.ndarray:
    """Compute MT ratio (signal with MT / signal without MT).

    Simplified Ramani model: the bound pool absorbs RF energy
    proportional to its lineshape at the offset frequency.

    Parameters
    ----------
    F : bound pool fraction.
    kf : forward exchange rate (s^-1).
    T2b : bound pool T2 (s).
    R1f : free pool R1 = 1/T1 of free water (s^-1).
    offset_freq : (N,) MT offset frequencies in Hz.
    mt_flip_angle : (N,) effective flip angle of MT pulse (radians).
    mt_pulse_duration : float — MT pulse duration (s).
    """
    # Super-Lorentzian lineshape for bound pool (typical for WM)
    # G(Δ) ≈ T2b / (π * (1 + (2π·Δ·T2b)^2))  (Lorentzian approximation)
    G = T2b / (jnp.pi * (1 + (2 * jnp.pi * offset_freq * T2b) ** 2))

    # Saturation rate of bound pool
    # W = π * (B1_rms)^2 * G(Δ)  where B1_rms depends on mt_flip_angle and duration
    # Simplified: W ∝ (flip_angle / duration)^2 * G
    omega1_sq = (mt_flip_angle / mt_pulse_duration) ** 2
    W = jnp.pi * omega1_sq * G

    # Steady-state signal ratio (Ramani Eq. 12)
    # MTR = 1 - Mz_free / M0_free
    # Mz_free = M0*(R1f*kf + R1f*W + kf*W) / (R1f*kf + R1f*W + kf*W + F*kf*R1f/(1-F))
    R1b = R1f  # assume R1b ≈ R1f (common simplification)
    kb = kf * F / jnp.maximum(1 - F, 1e-6)

    num = R1f + kf + W
    den = R1f + kf + W + kb * W / jnp.maximum(R1b + kb + W, 1e-6)

    signal_ratio = num / jnp.maximum(den, 1e-10)
    return jnp.clip(signal_ratio, 0.0, 1.0)


def fit_qmt(
    mt_on_data: jnp.ndarray,
    mt_off_data: jnp.ndarray,
    offset_freq: jnp.ndarray,
    mt_flip_angle: jnp.ndarray,
    R1f: float = 1.0,
    mt_pulse_duration: float = 0.01,
    n_iter: int = 40,
) -> QMTResult:
    """Fit qMT two-pool model with Laplace uncertainty.

    Parameters
    ----------
    mt_on_data : (N,) signal with MT pulse at each offset/flip.
    mt_off_data : (N,) or scalar — signal without MT (reference).
    offset_freq : (N,) MT offset frequencies (Hz).
    mt_flip_angle : (N,) MT pulse flip angles (radians).
    R1f : float — free pool R1 (from VFA T1 fitting), default 1 s^-1.
    """
    # Data as ratio
    mt_off = jnp.mean(mt_off_data) if mt_off_data.ndim > 0 and mt_off_data.shape[0] > 1 else mt_off_data
    data_ratio = mt_on_data / jnp.maximum(mt_off, 1e-10)
    data_ratio = jnp.clip(data_ratio, 0.0, 1.0)

    # Grid search initialization
    F_grid = jnp.array([0.05, 0.08, 0.12, 0.16, 0.20])
    kf_grid = jnp.array([1.0, 2.0, 4.0, 8.0])
    T2b_init = 10e-6  # 10 microseconds (typical for bound pool)

    def eval_grid(F_val, kf_val):
        pred = qmt_signal_ratio(F_val, kf_val, T2b_init, R1f, offset_freq,
                                 mt_flip_angle, mt_pulse_duration)
        return jnp.mean((pred - data_ratio) ** 2)

    losses = jax.vmap(jax.vmap(eval_grid, in_axes=(None, 0)), in_axes=(0, None))(F_grid, kf_grid)
    best_idx = jnp.unravel_index(jnp.argmin(losses), losses.shape)

    params = jnp.array([F_grid[best_idx[0]], kf_grid[best_idx[1]], T2b_init])

    # Loss function
    def loss_fn(params):
        F = jnp.clip(params[0], 0.01, 0.4)
        kf = jnp.maximum(params[1], 0.1)
        T2b = jnp.maximum(params[2], 1e-7)
        pred = qmt_signal_ratio(F, kf, T2b, R1f, offset_freq, mt_flip_angle, mt_pulse_duration)
        return jnp.mean((pred - data_ratio) ** 2)

    # Gauss-Newton refinement
    def gn_step(params, _):
        g = jax.grad(loss_fn)(params)
        H = jax.hessian(loss_fn)(params)
        damping = jnp.diag(jnp.abs(jnp.diag(H)) * 0.05 + 1e-8)
        step = jnp.linalg.solve(H + damping, g)
        new_params = params - step * 0.3  # conservative
        new_params = new_params.at[0].set(jnp.clip(new_params[0], 0.01, 0.4))
        new_params = new_params.at[1].set(jnp.clip(new_params[1], 0.1, 20.0))
        new_params = new_params.at[2].set(jnp.clip(new_params[2], 1e-7, 1e-4))
        return new_params, None

    params, _ = jax.lax.scan(gn_step, params, None, length=n_iter)

    pred = qmt_signal_ratio(params[0], params[1], params[2], R1f,
                             offset_freq, mt_flip_angle, mt_pulse_duration)
    residual = jnp.mean((pred - data_ratio) ** 2)

    # Uncertainty
    N = data_ratio.shape[0]
    sigma_sq = jnp.maximum(jnp.sum((pred - data_ratio) ** 2) / jnp.maximum(N - 3, 1), 1e-8)
    H = jax.hessian(loss_fn)(params)
    cov = sigma_sq * jnp.linalg.inv(H + 1e-8 * jnp.eye(3))

    F_std = jnp.sqrt(jnp.maximum(cov[0, 0], 0.0))
    kf_std = jnp.sqrt(jnp.maximum(cov[1, 1], 0.0))

    return QMTResult(F=params[0], kf=params[1], T2b=params[2],
                     F_std=F_std, kf_std=kf_std, residual=residual)
