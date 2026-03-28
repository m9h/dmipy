"""MP2RAGE T1 mapping — JAX implementation with uncertainty.

Computes quantitative T1 from two inversion images via the
uniform (B1-insensitive) contrast and a lookup table.

Reference: Marques JP et al. (2010) MP2RAGE, a self bias-field
corrected sequence for improved segmentation and T1-mapping at
high field. NeuroImage 49(2):1271-1281.
"""

from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


@jax.jit
def mp2rage_uniform(INV1: jnp.ndarray, INV2: jnp.ndarray) -> jnp.ndarray:
    """Compute the MP2RAGE uniform image (B1-insensitive).

    UNI = real(INV1 * conj(INV2)) / (INV1^2 + INV2^2 + eps)
    Bounded to [-0.5, 0.5].
    """
    num = INV1 * INV2
    den = INV1 ** 2 + INV2 ** 2 + 1e-10
    uni = num / den
    return jnp.clip(uni, -0.5, 0.5)


@jax.jit
def mp2rage_signal_model(
    T1: jnp.ndarray,
    TI1: float, TI2: float, TR: float,
    alpha1: float, alpha2: float,
    n_slices: int = 160,
    efficiency: float = 0.96,
) -> jnp.ndarray:
    """Simplified MP2RAGE signal: compute UNI value as function of T1.

    Uses the steady-state FLASH signal at each inversion time.
    """
    # Time between inversions spent on FLASH readout
    TA = TI1  # time from inversion to first readout
    TB = TI2 - TI1  # time between readouts
    TC = TR - TI2  # time after second readout

    E1_A = jnp.exp(-TA / T1)
    E1_B = jnp.exp(-TB / T1)
    E1_C = jnp.exp(-TC / T1)

    cos_a1 = jnp.cos(alpha1)
    cos_a2 = jnp.cos(alpha2)

    # Steady-state magnetization at each inversion time (simplified)
    # After inversion: M = -efficiency * M0
    # After recovery TA: M1 ≈ 1 - (1 + efficiency) * E1_A
    M1 = 1 - (1 + efficiency) * E1_A
    M2 = 1 - (1 + efficiency) * E1_A * E1_B

    # FLASH signal at each TI
    S1 = jnp.sin(alpha1) * M1
    S2 = jnp.sin(alpha2) * M2

    # Uniform image
    uni = S1 * S2 / (S1 ** 2 + S2 ** 2 + 1e-10)
    return jnp.clip(uni, -0.5, 0.5)


class MP2RAGEResult(NamedTuple):
    T1: float
    T1_std: float
    uniform: float


def fit_mp2rage_t1(
    INV1: float,
    INV2: float,
    TI1: float = 0.7,
    TI2: float = 2.5,
    TR: float = 5.0,
    alpha1: float = 4 * jnp.pi / 180,
    alpha2: float = 5 * jnp.pi / 180,
    n_slices: int = 160,
    efficiency: float = 0.96,
    n_lookup: int = 1000,
) -> MP2RAGEResult:
    """Fit T1 from MP2RAGE via lookup table with uncertainty."""
    # Compute uniform value
    uni = mp2rage_uniform(INV1, INV2)

    # Build lookup table: T1 → UNI
    T1_range = jnp.linspace(0.1, 6.0, n_lookup)
    uni_lookup = jax.vmap(
        lambda t1: mp2rage_signal_model(t1, TI1, TI2, TR, alpha1, alpha2, n_slices, efficiency)
    )(T1_range)

    # Find T1 by nearest lookup (monotonic relationship assumed)
    # Find closest match
    dist = jnp.abs(uni_lookup - uni)
    idx = jnp.argmin(dist)
    T1_fit = T1_range[idx]

    # Uncertainty via delta method: dT1 = dUNI / |dUNI/dT1|
    # Numerical derivative of the lookup
    duni_dt1 = jnp.gradient(uni_lookup, T1_range[1] - T1_range[0])
    slope_at_fit = jnp.abs(duni_dt1[idx])

    # Noise on UNI propagated from INV1/INV2
    # Assuming Gaussian noise: sigma_UNI ≈ sqrt(partial derivs)
    # Simplified: use the residual between measured and lookup UNI
    uni_residual = jnp.abs(uni - uni_lookup[idx])
    sigma_uni = jnp.maximum(uni_residual, 0.001)  # floor for noiseless case
    T1_std = sigma_uni / jnp.maximum(slope_at_fit, 1e-6)

    return MP2RAGEResult(T1=T1_fit, T1_std=T1_std, uniform=uni)


fit_mp2rage_t1_batch = jax.vmap(fit_mp2rage_t1, in_axes=(0, 0, None, None, None, None, None, None, None))
