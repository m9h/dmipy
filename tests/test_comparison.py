"""
Tests for the comparison runner framework.
"""

import numpy as np
import pytest


class TestComparisonResult:

    def test_add_method(self):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        result = ComparisonResult(vol_shape=(5, 5, 5))
        result.add_method("A", {"FA": np.random.rand(5, 5, 5)})
        result.add_method("B", {"FA": np.random.rand(5, 5, 5)})
        assert "A" in result.methods
        assert "B" in result.methods

    def test_metrics_available(self):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        result = ComparisonResult()
        result.add_method("A", {"FA": np.zeros(10), "MD": np.zeros(10)})
        result.add_method("B", {"FA": np.zeros(10), "RD": np.zeros(10)})
        # Only FA is common
        assert result.metrics_available() == ["FA"]

    def test_correlation(self):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        rng = np.random.RandomState(42)
        fa = rng.rand(100)
        result = ComparisonResult()
        result.add_method("A", {"FA": fa})
        result.add_method("B", {"FA": fa + rng.randn(100) * 0.01})
        r = result.correlation("A", "B", "FA")
        assert r > 0.99

    def test_rmse_against_gt(self):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        gt = {"FA": np.array([0.5, 0.6, 0.7])}
        result = ComparisonResult(ground_truth=gt)
        result.add_method("A", {"FA": np.array([0.5, 0.6, 0.7])})
        assert result.rmse("A", "FA") < 1e-10

    def test_ssim_perfect(self):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        x = np.random.rand(50)
        result = ComparisonResult(ground_truth={"FA": x})
        result.add_method("A", {"FA": x.copy()})
        assert result.ssim_1d("A", "FA") > 0.999

    def test_print_summary_runs(self, capsys):
        from dmipy_jax.pipeline.comparison import ComparisonResult

        rng = np.random.RandomState(0)
        gt = {"FA": rng.rand(20), "MD": rng.rand(20)}
        result = ComparisonResult(ground_truth=gt)
        result.add_method("A", {"FA": rng.rand(20), "MD": rng.rand(20)})
        result.add_method("B", {"FA": rng.rand(20), "MD": rng.rand(20)})
        result.print_summary()
        out = capsys.readouterr().out
        assert "Ground Truth" in out
        assert "Correlations" in out


class TestDipyDTI:

    def test_run_dipy_dti(self):
        """DIPY DTI on simple isotropic data should give FA ~ 0."""
        from dipy.sims.voxel import multi_tensor
        from dipy.core.gradients import gradient_table
        from dmipy_jax.pipeline.comparison import run_dipy_dti

        rng = np.random.RandomState(42)
        bvecs = rng.randn(30, 3)
        bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvals = np.concatenate([np.zeros(2), np.full(30, 1000)])
        bvecs = np.concatenate([np.array([[1, 0, 0], [0, 1, 0]]), bvecs])

        gtab = gradient_table(bvals, bvecs)
        evals = np.array([1e-3, 1e-3, 1e-3])
        signal, _ = multi_tensor(gtab, [evals], S0=1.0, angles=[(0, 0)],
                                  fractions=[100], snr=None)

        data = np.tile(signal, (3, 3, 1, 1))  # (3, 3, 1, 32)
        maps = run_dipy_dti(data, bvals, bvecs)
        assert "FA" in maps
        assert "MD" in maps
        assert np.all(maps["FA"] < 0.1)  # isotropic


class TestComparisonRunner:

    def test_runner_with_dipy(self):
        """ComparisonRunner with just DIPY DTI."""
        from dmipy_jax.pipeline.comparison import (
            ComparisonRunner, generate_dipy_multitensor_phantom,
        )

        data, bvals, bvecs, mask, gt = generate_dipy_multitensor_phantom(
            vol_shape=(6, 6, 1), snr=100,
        )
        runner = ComparisonRunner()
        runner.add_dipy_dti()
        result = runner.run(data, bvals, bvecs, mask=mask,
                            ground_truth=gt, bvals_unit="fsl")
        assert "DIPY DTI" in result.methods
        assert "FA" in result.methods["DIPY DTI"]
        # FA should be correlated with ground truth
        r = result.correlation("DIPY DTI", "DIPY DTI", "FA")
        assert r > 0.99  # self-correlation


class TestPhantomGeneration:

    def test_dipy_multitensor_phantom_shapes(self):
        from dmipy_jax.pipeline.comparison import generate_dipy_multitensor_phantom

        data, bvals, bvecs, mask, gt = generate_dipy_multitensor_phantom(
            vol_shape=(6, 6, 2), snr=100,
        )
        assert data.shape == (6, 6, 2, 32)
        assert bvals.shape == (32,)
        assert bvecs.shape == (32, 3)
        assert mask.shape == (6, 6, 2)
        assert gt["FA"].shape == (6, 6, 2)
        assert gt["MD"].shape == (6, 6, 2)

    def test_phantom_wm_has_high_fa(self):
        from dmipy_jax.pipeline.comparison import generate_dipy_multitensor_phantom

        data, bvals, bvecs, mask, gt = generate_dipy_multitensor_phantom(
            vol_shape=(9, 3, 1), snr=100,
        )
        # WM is x < 3
        wm_fa = gt["FA"][:3, :, :]
        assert np.mean(wm_fa) > 0.5

    def test_phantom_csf_has_low_fa(self):
        from dmipy_jax.pipeline.comparison import generate_dipy_multitensor_phantom

        data, bvals, bvecs, mask, gt = generate_dipy_multitensor_phantom(
            vol_shape=(9, 3, 1), snr=100,
        )
        # CSF is x >= 6
        csf_fa = gt["FA"][6:, :, :]
        assert np.mean(csf_fa) < 0.1
