"""
Analytical metric derivation from multi-compartment parameters.

Implements the closed-form formulas from FORCE Appendix A for computing
DTI, DKI, and NODDI metrics directly from microstructural parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# --------------------------------------------------------------------------- #
# DTI metrics
# --------------------------------------------------------------------------- #

def params_to_dti_tensor(
    eigenvalues: Float[Array, "... 3"],
) -> Float[Array, "... 3 3"]:
    """Build a diagonal diffusion tensor from eigenvalues.

    Parameters
    ----------
    eigenvalues : ``(..., 3)``
        ``[lambda_1, lambda_2, lambda_3]`` in descending order.

    Returns
    -------
    tensor : ``(..., 3, 3)``
        Diagonal diffusion tensor(s).
    """
    return jnp.apply_along_axis(jnp.diag, -1, eigenvalues)


def tensor_to_fa_md(
    eigenvalues: Float[Array, "... 3"],
) -> tuple:
    """Compute FA and MD from tensor eigenvalues.

    Parameters
    ----------
    eigenvalues : ``(..., 3)``

    Returns
    -------
    fa : ``(...)``
    md : ``(...)``
    """
    l1, l2, l3 = eigenvalues[..., 0], eigenvalues[..., 1], eigenvalues[..., 2]
    md = (l1 + l2 + l3) / 3.0
    num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
    denom = l1 ** 2 + l2 ** 2 + l3 ** 2
    eps = jnp.maximum(denom * 1e-10, 1e-30)
    fa = jnp.sqrt(jnp.clip(1.5 * num / (denom + eps), 0.0, 1.0))
    return fa, md


def eigenvalues_to_ad_rd(
    eigenvalues: Float[Array, "... 3"],
) -> tuple:
    """Compute axial and radial diffusivity.

    Returns
    -------
    ad : ``(...)`` – axial diffusivity (lambda_1)
    rd : ``(...)`` – radial diffusivity  mean(lambda_2, lambda_3)
    """
    ad = eigenvalues[..., 0]
    rd = (eigenvalues[..., 1] + eigenvalues[..., 2]) / 2.0
    return ad, rd


# --------------------------------------------------------------------------- #
# Multi-compartment → effective tensor
# --------------------------------------------------------------------------- #

def multi_compartment_to_eigenvalues(
    fractions: Float[Array, "... C"],
    d_pars: Float[Array, "... C"],
    d_perps: Float[Array, "... C"],
) -> Float[Array, "... 3"]:
    """Approximate effective eigenvalues of a mixture of axially-symmetric compartments.

    Uses the volume-weighted average (FORCE Eq. A1).

    Parameters
    ----------
    fractions : ``(..., C)``  compartment fractions (sum = 1).
    d_pars    : ``(..., C)``  parallel diffusivities.
    d_perps   : ``(..., C)``  perpendicular diffusivities.

    Returns
    -------
    eigenvalues : ``(..., 3)``  ``[lambda_1, lambda_2, lambda_3]``.
    """
    eff_par = jnp.sum(fractions * d_pars, axis=-1)
    eff_perp = jnp.sum(fractions * d_perps, axis=-1)
    return jnp.stack([eff_par, eff_perp, eff_perp], axis=-1)


# --------------------------------------------------------------------------- #
# DKI metrics (kurtosis via variance formula)
# --------------------------------------------------------------------------- #

def multi_compartment_to_mean_kurtosis(
    fractions: Float[Array, "... C"],
    d_pars: Float[Array, "... C"],
    d_perps: Float[Array, "... C"],
) -> Float[Array, "..."]:
    """Compute mean kurtosis from multi-compartment params.

    Uses the variance-of-means formula:
        MK = 3 * Var_c[D_c] / <D>^2

    where the average is weighted by compartment fractions.
    """
    # Mean diffusivity per compartment
    d_mean_c = (d_pars + 2 * d_perps) / 3.0  # (..., C)

    # Weighted mean
    d_bar = jnp.sum(fractions * d_mean_c, axis=-1)  # (...)

    # Weighted variance
    var_d = jnp.sum(fractions * (d_mean_c - d_bar[..., None]) ** 2, axis=-1)

    mk = 3.0 * var_d / (d_bar ** 2 + 1e-20)
    return mk


def multi_compartment_to_axial_kurtosis(
    fractions: Float[Array, "... C"],
    d_pars: Float[Array, "... C"],
) -> Float[Array, "..."]:
    """Axial kurtosis from multi-compartment parallel diffusivities.

    ``AK = 3 * Var_c[d_par_c] / <d_par>^2``

    This is the kurtosis along the principal diffusion direction,
    measuring the heterogeneity of axial diffusivities across compartments.
    """
    d_bar = jnp.sum(fractions * d_pars, axis=-1)
    var_d = jnp.sum(fractions * (d_pars - d_bar[..., None]) ** 2, axis=-1)
    return 3.0 * var_d / (d_bar ** 2 + 1e-20)


def multi_compartment_to_radial_kurtosis(
    fractions: Float[Array, "... C"],
    d_perps: Float[Array, "... C"],
) -> Float[Array, "..."]:
    """Radial kurtosis from multi-compartment perpendicular diffusivities.

    ``RK = 3 * Var_c[d_perp_c] / <d_perp>^2``

    Kurtosis perpendicular to the principal diffusion direction.
    """
    d_bar = jnp.sum(fractions * d_perps, axis=-1)
    var_d = jnp.sum(fractions * (d_perps - d_bar[..., None]) ** 2, axis=-1)
    return 3.0 * var_d / (d_bar ** 2 + 1e-20)


def multi_compartment_to_kurtosis_fa(
    fractions: Float[Array, "... C"],
    d_pars: Float[Array, "... C"],
    d_perps: Float[Array, "... C"],
) -> Float[Array, "..."]:
    """Kurtosis Fractional Anisotropy (KFA) from multi-compartment params.

    KFA is defined analogously to FA but applied to the kurtosis tensor:

    ``KFA = sqrt( (AK - MK)^2 + 2*(RK - MK)^2 ) / sqrt( AK^2 + 2*RK^2 )``

    This measures the anisotropy of the kurtosis tensor itself — high values
    indicate that kurtosis depends strongly on direction.
    """
    mk = multi_compartment_to_mean_kurtosis(fractions, d_pars, d_perps)
    ak = multi_compartment_to_axial_kurtosis(fractions, d_pars)
    rk = multi_compartment_to_radial_kurtosis(fractions, d_perps)

    num = (ak - mk) ** 2 + 2.0 * (rk - mk) ** 2
    denom = ak ** 2 + 2.0 * rk ** 2
    eps = jnp.maximum(denom * 1e-10, 1e-30)
    return jnp.sqrt(jnp.clip(num / (denom + eps), 0.0, 1.0))


def micro_fa(
    fractions: Float[Array, "... C"],
    d_pars: Float[Array, "... C"],
    d_perps: Float[Array, "... C"],
) -> Float[Array, "..."]:
    """Microscopic FA (micro-FA / uFA) from multi-compartment params.

    Computed as the volume-weighted average of per-compartment FA values.
    This captures anisotropy at the compartment level, independent of
    orientation dispersion.
    """
    # Per-compartment eigenvalues (axially symmetric: d_par, d_perp, d_perp)
    eigs_c = jnp.stack([d_pars, d_perps, d_perps], axis=-1)  # (..., C, 3)
    fa_c, _ = tensor_to_fa_md(eigs_c)  # (..., C)
    return jnp.sum(fractions * fa_c, axis=-1)


# --------------------------------------------------------------------------- #
# NODDI metrics
# --------------------------------------------------------------------------- #

def watson_kappa_to_odi(kappa: Float[Array, "..."]) -> Float[Array, "..."]:
    """Convert Watson concentration kappa to Orientation Dispersion Index.

    ``ODI = (2/pi) * arctan(1/kappa)``
    """
    return (2.0 / jnp.pi) * jnp.arctan(1.0 / (kappa + 1e-8))


def params_to_noddi_odi(
    kappa: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Alias for :func:`watson_kappa_to_odi`."""
    return watson_kappa_to_odi(kappa)


def bingham_kappas_to_odi(
    kappa1: Float[Array, "..."],
    kappa2: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Orientation dispersion index for Bingham distribution.

    Uses the geometric mean of the two dispersion axes::

        ODI = (2/pi) * arctan(1 / sqrt(kappa1 * kappa2))
    """
    kappa_eff = jnp.sqrt(kappa1 * kappa2 + 1e-12)
    return (2.0 / jnp.pi) * jnp.arctan(1.0 / (kappa_eff + 1e-8))


# --------------------------------------------------------------------------- #
# Fiber peak extraction
# --------------------------------------------------------------------------- #

def extract_fiber_peaks(
    orientations: Float[Array, "... F 3"],
    fractions: Float[Array, "... F"],
    threshold: float = 0.05,
) -> Float[Array, "... F 3"]:
    """Return fiber peaks above a volume-fraction threshold.

    Peaks below *threshold* are zeroed out.
    """
    mask = fractions > threshold
    return orientations * mask[..., None]
