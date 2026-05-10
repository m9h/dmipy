"""
Integration tests pinning the FORCE → DiSCo connectivity-matrix pipeline.

This reproduces the FORCE paper §3.2 protocol literally:
  1. Fit FORCE on DiSCo single-shell b=1900 with the tuned library.
  2. Extract peaks via force_peaks(fit) → PeaksAndMetrics.
  3. Run LocalTracking (EuDX-style) on the peaks within the ROI mask.
  4. Compute connectivity_matrix(streamlines, ROI labels).
  5. Compare to ground-truth Connectivity_Matrix_Cross-Sectional_Area.txt.

Each test pins one stage of the pipeline.
"""

from pathlib import Path

import numpy as np
import pytest


conn = pytest.importorskip("dmipy_jax.validation.force_disco_connectivity")


DISCO_ROOT = Path.home() / ".dipy" / "disco" / "disco_1"


@pytest.fixture(scope="module")
def disco_ready():
    if not DISCO_ROOT.exists():
        pytest.skip(f"DiSCo not at {DISCO_ROOT}")


# --------------------------------------------------------------------------- #
# 1. Ground-truth connectivity matrix loader
# --------------------------------------------------------------------------- #

class TestGroundTruthConnectivity:
    def test_load_gt_returns_16x16(self, disco_ready):
        gt = conn.load_gt_connectivity(subject=1)
        assert gt.shape == (16, 16)
        # Cross-sectional areas are nonnegative
        assert (gt >= 0).all()
        # Has 50 nonzero entries per DiSCo subject 1 spec
        assert (gt > 0).sum() == 50


# --------------------------------------------------------------------------- #
# 2. Connectivity matrix Pearson r
# --------------------------------------------------------------------------- #

class TestConnectivityPearson:
    def test_self_correlation_is_one(self):
        rng = np.random.default_rng(0)
        m = rng.exponential(size=(16, 16))
        m = m + m.T  # symmetric
        assert conn.connectivity_pearson(m, m) == pytest.approx(1.0, abs=1e-6)

    def test_zero_matrix_returns_nan(self):
        m = np.zeros((16, 16))
        assert np.isnan(conn.connectivity_pearson(m, m))

    def test_uses_upper_triangle_only(self):
        """Both matrices should be reduced to their symmetric (upper-tri,
        excluding diagonal) entries before correlation, to avoid double-
        counting symmetric pairs and the trivial-zero diagonal."""
        rng = np.random.default_rng(7)
        a = rng.exponential(size=(16, 16)); a = a + a.T
        b = a.copy()
        # Perturb only the lower triangle (which we'll discard) — pearson
        # should still be 1.0
        b[np.tril_indices(16, k=-1)] += rng.normal(scale=10, size=16 * 15 // 2)
        r = conn.connectivity_pearson(a, b)
        assert r == pytest.approx(1.0, abs=1e-6), (
            f"Function appears to use full matrix not upper-triangle; r={r}"
        )


# --------------------------------------------------------------------------- #
# 3. Pipeline smoke: fit a single ROI, get peaks, tract, count streamlines
# --------------------------------------------------------------------------- #

class TestTractographySmoke:
    def test_streamline_generation_returns_nonempty_default_lib(self, disco_ready):
        """End-to-end smoke: load DiSCo, refit FORCE on a 3-ROI subset,
        run tractography on the default-library fit. Must produce at
        least one streamline.
        """
        result = conn.run_force_connectivity(
            subject=1, snr=30, tuned=False,
            _smoke_roi_subset=(1, 2, 3), seed_density=1,
        )
        assert result["streamlines_count"] > 0
        assert result["connectivity"].shape == (16, 16)

    def test_tuned_library_suppresses_odfs_upstream_bug(self, disco_ready):
        """**Regression test for an upstream dipy bug**: tuned library
        generated with ``wm_threshold=1.0`` has all-zero ODFs, so
        ``force_peaks`` returns zero peaks despite ``num_fibers > 0``.

        Tractography on a tuned-library FORCEFit therefore produces
        zero streamlines. This test pins that behaviour so we'll know
        if/when dipy fixes the upstream ODF computation.

        If this test ever STARTS failing (i.e. tuned lib starts
        producing streamlines), the upstream bug is fixed — at which
        point we should switch ``run_force_connectivity`` to use
        ``tuned=True`` for the paper-aligned protocol.
        """
        import numpy as np
        from dipy.reconst.force import load_force_simulations
        from pathlib import Path

        tuned_lib = Path.home() / ".cache" / "dipy_force" / "force_disco_500k_tuned.npz"
        if not tuned_lib.exists():
            pytest.skip("tuned DiSCo library not cached")
        sims = load_force_simulations(str(tuned_lib))
        nz_odfs = (np.abs(sims["odfs"]).sum(axis=1) > 0).sum()
        assert nz_odfs == 0, (
            f"Tuned library now has {nz_odfs} nonzero ODFs (was 0). "
            "Upstream bug likely fixed — switch run_force_connectivity "
            "to use tuned=True."
        )

        # Consequence: tractography on tuned library produces 0 streamlines
        result = conn.run_force_connectivity(
            subject=1, snr=30, tuned=True,
            _smoke_roi_subset=(1, 2, 3), seed_density=1,
        )
        assert result["streamlines_count"] == 0, (
            "Tuned-library tractography unexpectedly produced "
            f"{result['streamlines_count']} streamlines; ODF "
            "suppression bug appears fixed."
        )
