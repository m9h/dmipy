"""
FORCE → DiSCo connectivity matrix pipeline (reproduces FORCE paper §3.2).

Pipeline:
  1. Fit FORCE on a DiSCo subject's single-shell b=1900 data with the
     paper-protocol tuned library.
  2. Extract a PeaksAndMetrics object via force_peaks(fit).
  3. Run LocalTracking (Euler integration with a discrete peak direction
     getter — the EuDX-equivalent in current dipy).
  4. Filter streamlines to those connecting distinct ROI labels.
  5. Build a 16×16 connectivity matrix via
     dipy.tracking.utils.connectivity_matrix.
  6. Compute Pearson r against the ground-truth Connectivity_Matrix
     loaded from DiSCo1_Connectivity_Matrix_Cross-Sectional_Area.txt.

The paper §3.2 reports r=0.868 at SNR=10 and r=0.894 at SNR=50 on this
pipeline against the same ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_DISCO_ROOT = Path.home() / ".dipy" / "disco"


def _disco_path(subject: int = 1) -> Path:
    p = _DISCO_ROOT / f"disco_{subject}"
    if not p.exists():
        raise FileNotFoundError(
            f"DiSCo subject {subject} not at {p}; fetch with "
            f"dipy.data.fetch_disco{subject}_dataset()."
        )
    return p


def load_gt_connectivity(subject: int = 1) -> np.ndarray:
    """Load DiSCo ground-truth Connectivity_Matrix_Cross-Sectional_Area.

    Returns a 16×16 float matrix of bundle cross-sectional areas in mm².
    The diagonal is zero (no self-connections); the matrix is symmetric.
    """
    base = _disco_path(subject)
    gt = np.loadtxt(base / f"DiSCo{subject}_Connectivity_Matrix_Cross-Sectional_Area.txt")
    assert gt.shape == (16, 16), f"Unexpected GT matrix shape: {gt.shape}"
    return gt.astype(np.float64)


def connectivity_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two N×N connectivity matrices, on the upper
    triangle (excluding the diagonal).

    Both matrices are assumed symmetric. Using the upper triangle avoids
    double-counting symmetric pairs and excludes the trivial-zero
    diagonal entries.
    """
    iu = np.triu_indices(a.shape[0], k=1)
    av = np.asarray(a)[iu].astype(np.float64)
    bv = np.asarray(b)[iu].astype(np.float64)
    valid = np.isfinite(av) & np.isfinite(bv)
    if valid.sum() < 3 or av[valid].std() < 1e-12 or bv[valid].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(av[valid], bv[valid])[0, 1])


def run_force_connectivity(
    subject: int = 1,
    snr: int = 30,
    tuned: bool = True,
    seed_density: int = 2,
    step_size: float = 0.5,
    max_cross: int = 1,
    _smoke_roi_subset: Optional[Iterable[int]] = None,
):
    """Fit FORCE on DiSCo and produce a connectivity matrix.

    Parameters
    ----------
    seed_density
        Per-dimension density of seeds inside the ROI mask. The dipy
        convention is to seed ``density**3`` points per voxel. 2 → 8
        seeds/voxel.
    step_size
        Streamline integration step in voxels (mm in DiSCo's affine).
    max_cross
        Maximum number of fibres to follow per voxel from the FORCE
        peaks.
    _smoke_roi_subset
        For tests only: restrict seeding to the listed ROI labels so
        the smoke test produces a tiny streamline set quickly.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    import nibabel as nib
    from dipy.direction.peaks import PeaksAndMetrics
    from dipy.reconst.force import FORCEModel, force_peaks, load_force_simulations
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    from dipy.tracking.utils import connectivity_matrix, seeds_from_mask

    from dmipy_jax.validation.force_disco import load_disco_subject

    out = load_disco_subject(subject=subject, snr=snr, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    gtab = out["gtab"]
    rois = out["rois"]
    affine = nib.load(_disco_path(subject) / f"highRes_DiSCo{subject}_DWI.nii.gz").affine

    # Library
    lib_cache = (Path.home() / ".cache" / "dipy_force" /
                 ("force_disco_500k_tuned.npz" if tuned
                  else "force_disco_500k.npz"))
    if not lib_cache.exists():
        raise FileNotFoundError(
            f"FORCE library not cached at {lib_cache}; "
            "run validate_force_disco_phantom.py {--tuned} first."
        )
    sims = load_force_simulations(str(lib_cache))
    model = FORCEModel(gtab, simulations=sims, n_neighbors=50)

    # Fit
    if _smoke_roi_subset is not None:
        fit_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        fit_mask = mask
    fit = model.fit(data, mask=fit_mask)
    peaks = force_peaks(fit)

    # Stopping criterion: stay within brain mask
    sc = BinaryStoppingCriterion(mask.astype(np.uint8))

    # Seed from ROI mask (or smoke subset)
    if _smoke_roi_subset is not None:
        seed_mask = np.isin(rois, list(_smoke_roi_subset)) & mask
    else:
        seed_mask = (rois > 0) & mask
    seeds = seeds_from_mask(seed_mask, affine, density=seed_density)

    # Track
    streamline_gen = LocalTracking(
        peaks, sc, seeds, affine, step_size=step_size,
        max_cross=max_cross, return_all=False, maxlen=300, minlen=4,
    )
    streamlines = Streamlines(streamline_gen)
    # Drop any zero-length streamlines that LocalTracking can occasionally
    # emit (they crash dipy.tracking.utils.connectivity_matrix).
    streamlines = Streamlines([s for s in streamlines if len(s) >= 2])

    # Connectivity matrix using ROI labels (skip background 0)
    if len(streamlines) == 0:
        cmat = np.zeros((17, 17), dtype=np.int64)
    else:
        cmat, _ = connectivity_matrix(
            streamlines, affine, rois.astype(np.int32),
            return_mapping=True, mapping_as_streamlines=False, symmetric=True,
        )
    # cmat is (n_labels+1) × (n_labels+1) including background label 0;
    # the paper's matrix is 16×16 (labels 1..16).
    cmat16 = cmat[1:17, 1:17].astype(np.float64)

    return {
        "streamlines_count": len(streamlines),
        "connectivity": cmat16,
        "fit_mask_voxels": int(fit_mask.sum()),
        "n_seeds": int(len(seeds)),
    }
