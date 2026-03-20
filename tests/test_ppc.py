"""
Tests for Posterior Predictive Checks (PPC) diagnostics.
"""

import numpy as np
import pytest

from dmipy_jax.pipeline.ppc import PPCResult


class TestPPCResultProperties:
    """Verify PPCResult summary statistics are computed correctly."""

    def _make_result(self, n_checks=100, n_meas=10):
        rng = np.random.default_rng(42)
        return PPCResult(
            rmse_per_obs=rng.uniform(0.01, 0.1, size=(n_checks,)),
            coverage_per_measurement=rng.uniform(0.8, 1.0, size=(n_meas,)),
            chi_squared=rng.uniform(5, 15, size=(n_checks,)),
            n_measurements=n_meas,
        )

    def test_mean_rmse(self):
        result = self._make_result()
        expected = float(np.mean(result.rmse_per_obs))
        assert abs(result.mean_rmse - expected) < 1e-10

    def test_median_rmse(self):
        result = self._make_result()
        expected = float(np.median(result.rmse_per_obs))
        assert abs(result.median_rmse - expected) < 1e-10

    def test_mean_coverage(self):
        result = self._make_result()
        expected = float(np.mean(result.coverage_per_measurement))
        assert abs(result.mean_coverage - expected) < 1e-10

    def test_reduced_chi_squared(self):
        result = self._make_result()
        expected = float(np.mean(result.chi_squared)) / result.n_measurements
        assert abs(result.reduced_chi_squared - expected) < 1e-10

    def test_print_summary_runs(self, capsys):
        """print_summary should execute without error."""
        result = self._make_result()
        result.print_summary()
        captured = capsys.readouterr()
        assert "Posterior Predictive Check Results" in captured.out
        assert "Mean RMSE" in captured.out
        assert "Reduced chi-squared" in captured.out


class TestPerfectModelLowRMSE:
    """If posterior samples perfectly recover the true parameters, RMSE
    should be low (essentially zero in the noiseless case)."""

    def test_zero_rmse_when_predictions_match(self):
        """When posterior predictive mean equals x*, RMSE is 0."""
        n_checks = 50
        n_meas = 20
        rmse = np.zeros(n_checks)
        coverage = np.ones(n_meas)
        # chi-squared is 0 / var => 0 when residual is 0
        chi2 = np.zeros(n_checks)
        result = PPCResult(
            rmse_per_obs=rmse,
            coverage_per_measurement=coverage,
            chi_squared=chi2,
            n_measurements=n_meas,
        )
        assert result.mean_rmse == 0.0
        assert result.median_rmse == 0.0
        assert result.mean_coverage == 1.0
        assert result.reduced_chi_squared == 0.0

    def test_small_rmse_with_minor_noise(self):
        """Small residuals should yield small RMSE."""
        rng = np.random.default_rng(7)
        n_checks = 200
        n_meas = 30
        # Simulate small residuals
        rmse = rng.uniform(0.0, 0.01, size=(n_checks,))
        coverage = np.ones(n_meas)
        chi2 = rng.uniform(0.0, 0.5, size=(n_checks,))
        result = PPCResult(
            rmse_per_obs=rmse,
            coverage_per_measurement=coverage,
            chi_squared=chi2,
            n_measurements=n_meas,
        )
        assert result.mean_rmse < 0.01
        assert result.median_rmse < 0.01


class TestCoverageShape:
    """Verify coverage_per_measurement has the right shape."""

    def test_coverage_shape_matches_n_measurements(self):
        for n_meas in [1, 10, 50, 108]:
            result = PPCResult(
                rmse_per_obs=np.zeros(20),
                coverage_per_measurement=np.ones(n_meas) * 0.9,
                chi_squared=np.ones(20) * n_meas,
                n_measurements=n_meas,
            )
            assert result.coverage_per_measurement.shape == (n_meas,)

    def test_coverage_values_between_0_and_1(self):
        """Coverage fractions must be in [0, 1]."""
        rng = np.random.default_rng(123)
        n_meas = 25
        coverage = rng.uniform(0.0, 1.0, size=(n_meas,))
        result = PPCResult(
            rmse_per_obs=np.zeros(10),
            coverage_per_measurement=coverage,
            chi_squared=np.ones(10) * n_meas,
            n_measurements=n_meas,
        )
        assert np.all(result.coverage_per_measurement >= 0.0)
        assert np.all(result.coverage_per_measurement <= 1.0)


class TestReducedChiSquaredInterpretation:
    """Verify reduced chi-squared behaves as expected."""

    def test_ideal_chi_squared_near_one(self):
        """When chi2 ~ n_measurements for every obs, reduced chi2 ~ 1."""
        n_meas = 50
        n_checks = 300
        # Each chi2 is exactly n_measurements
        chi2 = np.full(n_checks, float(n_meas))
        result = PPCResult(
            rmse_per_obs=np.zeros(n_checks),
            coverage_per_measurement=np.ones(n_meas) * 0.9,
            chi_squared=chi2,
            n_measurements=n_meas,
        )
        assert abs(result.reduced_chi_squared - 1.0) < 1e-10

    def test_overfit_chi_squared_below_one(self):
        """If chi2 << n_measurements, model may be overfitting."""
        n_meas = 50
        chi2 = np.full(100, 5.0)  # much less than n_meas
        result = PPCResult(
            rmse_per_obs=np.zeros(100),
            coverage_per_measurement=np.ones(n_meas),
            chi_squared=chi2,
            n_measurements=n_meas,
        )
        assert result.reduced_chi_squared < 1.0
