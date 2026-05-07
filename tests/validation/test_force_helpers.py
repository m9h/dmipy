"""
Tests for the FORCE replication helper module.

These pin the behaviour of helpers extracted from
validation/validate_force_replication_v2.py — in particular the
``best_two_peaks`` regression where a dummy [0,0,1] fallback used to
falsely pass ``check_both_detected`` at narrow crossings.
"""

import math

import numpy as np
import pytest


# Import target — drives the red→green refactor of the v2 script's helpers
# into a real module.
force_helpers = pytest.importorskip("dmipy_jax.validation.force_helpers")


# --------------------------------------------------------------------------- #
# angular_error
# --------------------------------------------------------------------------- #

class TestAngularError:
    def test_aligned_is_zero(self):
        a = np.array([0.0, 0.0, 1.0])
        assert force_helpers.angular_error(a, a) == pytest.approx(0.0)

    def test_antipodal_is_zero(self):
        a = np.array([0.0, 0.0, 1.0])
        b = -a
        # antipodal vectors describe the same fibre orientation
        assert force_helpers.angular_error(a, b) == pytest.approx(0.0)

    def test_orthogonal_is_ninety(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        assert force_helpers.angular_error(a, b) == pytest.approx(90.0)

    def test_known_30_degrees(self):
        a = np.array([0.0, 0.0, 1.0])
        rad = math.radians(30)
        b = np.array([math.sin(rad), 0.0, math.cos(rad)])
        assert force_helpers.angular_error(a, b) == pytest.approx(30.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# best_two_peaks — regression target
# --------------------------------------------------------------------------- #

class TestBestTwoPeaks:
    def setup_method(self):
        self.mu1 = np.array([0.0, 0.0, 1.0])
        rad = math.radians(20)
        self.mu2 = np.array([math.sin(rad), 0.0, math.cos(rad)])

    def test_empty_returns_none(self):
        assert force_helpers.best_two_peaks([], self.mu1, self.mu2) is None

    def test_single_peak_returns_none(self):
        """The v1 bug: a single recovered peak used to be padded with a dummy
        [0,0,1] and falsely pass detection at narrow crossings. Must return
        None now."""
        only_one = [np.array([0.05, 0.0, 0.998])]
        assert force_helpers.best_two_peaks(only_one, self.mu1, self.mu2) is None

    def test_zero_norm_peaks_are_filtered(self):
        peaks = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.05, 0.0, 0.998]),
        ]
        assert force_helpers.best_two_peaks(peaks, self.mu1, self.mu2) is None

    def test_two_peaks_returns_pair(self):
        rad = math.radians(20)
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([math.sin(rad), 0.0, math.cos(rad)])
        result = force_helpers.best_two_peaks([a, b], self.mu1, self.mu2)
        assert result is not None
        assert len(result) == 2

    def test_three_peaks_picks_best_pair(self):
        rad = math.radians(20)
        good_a = np.array([0.0, 0.0, 1.0])
        good_b = np.array([math.sin(rad), 0.0, math.cos(rad)])
        decoy = np.array([1.0, 0.0, 0.0])  # 90° away — should be rejected
        result = force_helpers.best_two_peaks(
            [good_a, decoy, good_b], self.mu1, self.mu2
        )
        assert result is not None
        chosen_set = {tuple(np.round(r, 6).tolist()) for r in result}
        assert tuple(np.round(good_a, 6).tolist()) in chosen_set
        assert tuple(np.round(good_b, 6).tolist()) in chosen_set


# --------------------------------------------------------------------------- #
# check_both_detected — symmetric pairing
# --------------------------------------------------------------------------- #

class TestCheckBothDetected:
    def setup_method(self):
        self.mu1 = np.array([0.0, 0.0, 1.0])
        rad = math.radians(45)
        self.mu2 = np.array([math.sin(rad), 0.0, math.cos(rad)])

    def test_perfect_recovery_passes(self):
        assert force_helpers.check_both_detected(
            self.mu1, self.mu2, self.mu1, self.mu2, threshold=15.0
        )

    def test_swapped_assignment_passes(self):
        """Function should pick the better of the two label assignments."""
        assert force_helpers.check_both_detected(
            self.mu1, self.mu2, self.mu2, self.mu1, threshold=15.0
        )

    def test_one_correct_one_off_fails(self):
        far = np.array([0.0, 1.0, 0.0])  # 90° from both
        assert not force_helpers.check_both_detected(
            self.mu1, self.mu2, self.mu1, far, threshold=15.0
        )

    def test_above_threshold_fails(self):
        """Errors clearly above threshold must fail detection."""
        rad = math.radians(16)
        over = np.array([math.sin(rad), 0.0, math.cos(rad)])
        # over is 16° from mu1 — must fail at threshold 15°
        assert not force_helpers.check_both_detected(
            self.mu1, self.mu2, over, self.mu2, threshold=15.0
        )

    def test_below_threshold_passes(self):
        """Errors clearly below threshold must pass detection."""
        rad = math.radians(14)
        under = np.array([math.sin(rad), 0.0, math.cos(rad)])
        # under is 14° from mu1 — must pass at threshold 15°
        assert force_helpers.check_both_detected(
            self.mu1, self.mu2, under, self.mu2, threshold=15.0
        )


# --------------------------------------------------------------------------- #
# params_to_orientations — recovers the two fiber directions from a flat
# parameter vector [d_par, theta1, theta2, f1, f_iso].
# --------------------------------------------------------------------------- #

class TestParamsToOrientations:
    def test_zero_thetas_aligned_with_z(self):
        params = np.array([1.7e-9, 0.0, 0.0, 0.5, 0.0])
        mu1, mu2 = force_helpers.params_to_orientations(params)
        assert mu1 == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-6)
        assert mu2 == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-6)

    def test_orthogonal_pair(self):
        params = np.array([1.7e-9, math.pi / 2, 0.0, 0.5, 0.0])
        mu1, mu2 = force_helpers.params_to_orientations(params)
        # mu1 should now be along +x
        assert abs(mu1 @ np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
        assert mu2 == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-6)
