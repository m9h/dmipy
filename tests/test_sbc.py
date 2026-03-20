"""
Tests for Simulation-Based Calibration (SBC) diagnostics.
"""

import numpy as np
import pytest

from dmipy_jax.pipeline.sbc import SBCResult, compute_ranks


class TestSBCResultShapes:
    """Verify SBCResult has correct shapes and metadata."""

    def test_ranks_shape(self):
        n_sbc, n_params, L = 100, 3, 200
        ranks = np.random.randint(0, L + 1, size=(n_sbc, n_params))
        result = SBCResult(
            ranks=ranks,
            parameter_names=["p1", "p2", "p3"],
            n_posterior_samples=L,
        )
        assert result.ranks.shape == (n_sbc, n_params)
        assert len(result.parameter_names) == n_params
        assert result.n_posterior_samples == L

    def test_coverage_shape(self):
        n_sbc, n_params, L = 50, 2, 100
        ranks = np.random.randint(0, L + 1, size=(n_sbc, n_params))
        result = SBCResult(
            ranks=ranks,
            parameter_names=["a", "b"],
            n_posterior_samples=L,
        )
        cov = result.coverage_at(0.9)
        assert cov.shape == (n_params,)

    def test_ks_test_keys(self):
        n_sbc, L = 80, 50
        names = ["alpha", "beta"]
        ranks = np.random.randint(0, L + 1, size=(n_sbc, 2))
        result = SBCResult(
            ranks=ranks,
            parameter_names=names,
            n_posterior_samples=L,
        )
        ks = result.ks_test()
        assert set(ks.keys()) == set(names)
        for name in names:
            assert "statistic" in ks[name]
            assert "pvalue" in ks[name]


class TestUniformRanksForPerfectModel:
    """Create synthetic uniform ranks and verify KS test passes."""

    def test_uniform_ranks_pass_ks(self):
        """Truly uniform ranks should not be rejected by KS test."""
        rng = np.random.default_rng(42)
        n_sbc = 1000
        L = 200
        n_params = 3
        # Uniform integer ranks in [0, L]
        ranks = rng.integers(0, L + 1, size=(n_sbc, n_params))
        result = SBCResult(
            ranks=ranks,
            parameter_names=["x", "y", "z"],
            n_posterior_samples=L,
        )
        ks = result.ks_test()
        for name in result.parameter_names:
            assert ks[name]["pvalue"] > 0.05, (
                f"Uniform ranks should pass KS test but {name} failed "
                f"with p={ks[name]['pvalue']:.4f}"
            )

    def test_nonuniform_ranks_fail_ks(self):
        """Ranks concentrated at the edges should fail the KS test."""
        rng = np.random.default_rng(99)
        n_sbc = 1000
        L = 200
        # Bimodal ranks: pile up at 0 and L (overconfident posterior)
        choices = np.array([0, L])
        ranks = rng.choice(choices, size=(n_sbc, 1))
        result = SBCResult(
            ranks=ranks,
            parameter_names=["bad_param"],
            n_posterior_samples=L,
        )
        ks = result.ks_test()
        assert ks["bad_param"]["pvalue"] < 0.05, (
            "Non-uniform ranks should fail KS test"
        )


class TestCoverageComputation:
    """Verify coverage_at works correctly with known data."""

    def test_coverage_perfect_centering(self):
        """If all ranks are exactly at 50%, coverage@90% should be 1.0."""
        L = 200
        n_sbc = 100
        # Rank = L/2 => normalised = 0.5, which is inside [0.05, 0.95]
        ranks = np.full((n_sbc, 2), L // 2)
        result = SBCResult(
            ranks=ranks,
            parameter_names=["a", "b"],
            n_posterior_samples=L,
        )
        cov = result.coverage_at(0.9)
        np.testing.assert_allclose(cov, 1.0)

    def test_coverage_all_at_edge(self):
        """If all ranks are 0 (extreme), coverage@90% should be 0.0."""
        L = 200
        n_sbc = 100
        ranks = np.zeros((n_sbc, 1), dtype=int)
        result = SBCResult(
            ranks=ranks,
            parameter_names=["edge"],
            n_posterior_samples=L,
        )
        cov = result.coverage_at(0.9)
        assert cov[0] == 0.0, f"Expected 0 coverage but got {cov[0]}"

    def test_coverage_uniform_near_nominal(self):
        """Uniform ranks should yield coverage close to the nominal level."""
        rng = np.random.default_rng(7)
        L = 500
        n_sbc = 5000
        ranks = rng.integers(0, L + 1, size=(n_sbc, 2))
        result = SBCResult(
            ranks=ranks,
            parameter_names=["u", "v"],
            n_posterior_samples=L,
        )
        for level in [0.5, 0.8, 0.9, 0.95]:
            cov = result.coverage_at(level)
            for d in range(2):
                assert abs(cov[d] - level) < 0.05, (
                    f"coverage_at({level}) = {cov[d]:.3f}, "
                    f"expected close to {level}"
                )


class TestComputeRanks:
    """Test the compute_ranks utility function."""

    def test_basic_ranking(self):
        import jax.numpy as jnp

        theta_true = jnp.array([0.5, 0.3])
        posterior = jnp.array([
            [0.1, 0.1],
            [0.4, 0.5],
            [0.6, 0.2],
            [0.8, 0.4],
        ])
        ranks = compute_ranks(theta_true, posterior)
        # dim 0: samples < 0.5 are [0.1, 0.4] => rank 2
        # dim 1: samples < 0.3 are [0.1, 0.2] => rank 2
        np.testing.assert_array_equal(np.asarray(ranks), [2, 2])

    def test_all_below(self):
        import jax.numpy as jnp

        theta_true = jnp.array([10.0])
        posterior = jnp.array([[1.0], [2.0], [3.0]])
        ranks = compute_ranks(theta_true, posterior)
        np.testing.assert_array_equal(np.asarray(ranks), [3])

    def test_all_above(self):
        import jax.numpy as jnp

        theta_true = jnp.array([0.0])
        posterior = jnp.array([[1.0], [2.0], [3.0]])
        ranks = compute_ranks(theta_true, posterior)
        np.testing.assert_array_equal(np.asarray(ranks), [0])
