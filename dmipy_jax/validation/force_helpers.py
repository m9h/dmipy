"""
Reusable helpers for FORCE-style crossing-angle validation experiments.

Extracted from ``validation/validate_force_replication_v2.py`` so they can be
unit-tested independently. Pure-numpy except for :func:`acq_to_gtab` which
takes a :class:`JaxAcquisition`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def angular_error(mu_true: np.ndarray, mu_recovered: np.ndarray) -> float:
    """Minimum angle (degrees) between two unit vectors with antipodal symmetry.

    Diffusion fibres have no head/tail, so antipodal vectors describe the
    same orientation; the absolute value of the dot product captures that.
    """
    dot = np.abs(np.dot(mu_true, mu_recovered))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def params_to_orientations(
    params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the two fibre directions from a flat 2-stick parameter vector.

    Parameters
    ----------
    params : ``[d_par, theta1, theta2, f1, f_iso]``

    Returns
    -------
    mu1, mu2 : two unit vectors in the +x/+z plane
    """
    theta1 = float(params[1])
    theta2 = float(params[2])
    mu1 = np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    mu2 = np.array([np.sin(theta2), 0.0, np.cos(theta2)])
    return mu1, mu2


# --------------------------------------------------------------------------- #
# Detection scoring
# --------------------------------------------------------------------------- #

def check_both_detected(
    mu1_true: np.ndarray,
    mu2_true: np.ndarray,
    mu1_rec: np.ndarray,
    mu2_rec: np.ndarray,
    threshold: float,
) -> bool:
    """True iff both ground-truth fibres are within *threshold* (deg) of one
    of the recovered peaks. Picks the assignment minimising total error."""
    err_11 = angular_error(mu1_true, mu1_rec)
    err_12 = angular_error(mu1_true, mu2_rec)
    err_21 = angular_error(mu2_true, mu1_rec)
    err_22 = angular_error(mu2_true, mu2_rec)
    if err_11 + err_22 < err_12 + err_21:
        return err_11 < threshold and err_22 < threshold
    return err_12 < threshold and err_21 < threshold


def best_two_peaks(
    peak_dirs,
    mu1_true: np.ndarray,
    mu2_true: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pick the two recovered peaks that best match the ground-truth pair.

    Returns ``None`` if fewer than 2 nonzero peaks were recovered — the
    method has *failed* to detect a crossing and should be scored as a miss,
    not rescued by a dummy fallback. The earlier dummy-``[0,0,1]`` behaviour
    falsely passed :func:`check_both_detected` when the truth pair happened
    to align with z (commit 633a637 fix).
    """
    nz: List[np.ndarray] = [
        np.asarray(p) for p in peak_dirs if np.linalg.norm(p) > 1e-6
    ]
    if len(nz) < 2:
        return None
    best, best_score = None, np.inf
    for i in range(len(nz)):
        for j in range(i + 1, len(nz)):
            a = angular_error(mu1_true, nz[i]) + angular_error(mu2_true, nz[j])
            b = angular_error(mu1_true, nz[j]) + angular_error(mu2_true, nz[i])
            score = min(a, b)
            if score < best_score:
                best_score = score
                best = (nz[i], nz[j])
    return best


# --------------------------------------------------------------------------- #
# Acquisition adapters
# --------------------------------------------------------------------------- #

def acq_to_gtab(acq):
    """JaxAcquisition -> DIPY :class:`gradient_table`.

    JaxAcquisition stores b-values in SI (s/m^2). DIPY expects s/mm^2.
    Imported lazily so this module is usable without DIPY for the geometry
    helpers.
    """
    from dipy.core.gradients import gradient_table

    bvals = np.asarray(acq.bvalues) / 1e6
    bvecs = np.asarray(acq.gradient_directions)
    return gradient_table(bvals, bvecs=bvecs)
