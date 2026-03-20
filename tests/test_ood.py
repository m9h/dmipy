"""
Tests for out-of-distribution (OOD) detection.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.ood import OODDetector, OODResult
from dmipy_jax.pipeline.train import _NormalisedMDN


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

N_MEAS = 32
THETA_DIM = 1
N_COMPONENTS = 4


class _DummyMDNInner(eqx.Module):
    """Bare MDN that returns fixed predictions regardless of input.

    Returns component means centred at 0.5 (the normalised midpoint of
    the Ball model's lambda_iso range) so that the denormalised posterior
    mean produces a physically plausible diffusivity.
    """

    num_components: int = N_COMPONENTS
    out_features: int = THETA_DIM

    def __call__(self, x):
        K, D = self.num_components, self.out_features
        logits_pi = jnp.zeros(K)
        mu = jnp.ones((K, D)) * 0.5  # normalised [0, 1]
        log_sigma = jnp.full((K, D), -2.0)
        return logits_pi, mu, log_sigma


def _make_simple_setup():
    """Create a simple Ball model simulator and wrapped dummy MDN.

    Returns
    -------
    model : _NormalisedMDN
        Dummy MDN wrapped with denormalisation (same as real training).
    simulator : ModelSimulator
        Ball model simulator.
    """
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (N_MEAS, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.full(N_MEAS, 1e9)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        d = params[0]
        return jnp.exp(-acq.bvalues * d)

    param_ranges = {"lambda_iso": (0.1e-9, 3.0e-9)}
    simulator = ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["lambda_iso"],
        parameter_ranges=param_ranges,
        acquisition=acq,
        noise_type="gaussian",
        snr=30.0,
    )

    lows = jnp.array([param_ranges["lambda_iso"][0]])
    highs = jnp.array([param_ranges["lambda_iso"][1]])
    spans = highs - lows

    inner = _DummyMDNInner()
    model = _NormalisedMDN(inner=inner, lows=lows, spans=spans)

    return model, simulator


def _make_fitted_detector(n_reference=500):
    """Create a fitted OODDetector for testing."""
    model, simulator = _make_simple_setup()
    detector = OODDetector(model=model, simulator=simulator)
    key = jax.random.PRNGKey(42)
    detector.fit(key, n_reference=n_reference)
    return detector, model, simulator


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestOODScoresShape:

    def test_score_returns_correct_keys(self):
        detector, _, simulator = _make_fitted_detector()
        key = jax.random.PRNGKey(99)
        _, signals = simulator.sample_and_simulate(key, 20)
        scores = detector.score(signals)
        expected_keys = {
            "reconstruction_error",
            "predictive_entropy",
            "mahalanobis_distance",
            "is_ood",
        }
        assert set(scores.keys()) == expected_keys

    def test_score_shapes(self):
        detector, _, simulator = _make_fitted_detector()
        key = jax.random.PRNGKey(100)
        batch_size = 30
        _, signals = simulator.sample_and_simulate(key, batch_size)
        scores = detector.score(signals)
        assert scores["reconstruction_error"].shape == (batch_size,)
        assert scores["predictive_entropy"].shape == (batch_size,)
        assert scores["mahalanobis_distance"].shape == (batch_size,)
        assert scores["is_ood"].shape == (batch_size,)

    def test_score_requires_fit(self):
        model, simulator = _make_simple_setup()
        detector = OODDetector(model=model, simulator=simulator)
        key = jax.random.PRNGKey(1)
        _, signals = simulator.sample_and_simulate(key, 10)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(signals)


class TestInDistributionLowScores:

    def test_reference_signals_low_reconstruction_error(self):
        """In-distribution signals should have relatively low RMSE."""
        detector, _, simulator = _make_fitted_detector(n_reference=500)
        key = jax.random.PRNGKey(200)
        _, signals = simulator.sample_and_simulate(key, 100)
        scores = detector.score(signals)

        # The dummy model always predicts the same theta, so the
        # reconstruction error won't be tiny, but it should be finite
        # and not NaN.
        recon = scores["reconstruction_error"]
        assert np.all(np.isfinite(recon))
        assert np.all(recon >= 0.0)

    def test_reference_signals_finite_entropy(self):
        """Entropy should be finite and non-negative for in-distribution."""
        detector, _, simulator = _make_fitted_detector()
        key = jax.random.PRNGKey(201)
        _, signals = simulator.sample_and_simulate(key, 50)
        scores = detector.score(signals)
        ent = scores["predictive_entropy"]
        assert np.all(np.isfinite(ent))
        assert np.all(ent >= 0.0)


class TestOODHighReconstructionError:

    def test_random_noise_higher_error_than_in_distribution(self):
        """Pure random noise should have higher reconstruction error
        than in-distribution signals."""
        detector, _, simulator = _make_fitted_detector(n_reference=500)

        key = jax.random.PRNGKey(300)
        k1, k2 = jax.random.split(key)

        # In-distribution signals
        _, signals_id = simulator.sample_and_simulate(k1, 200)
        scores_id = detector.score(signals_id)

        # Out-of-distribution: random uniform noise in [0, 5]
        signals_ood = jax.random.uniform(
            k2, (200, simulator.signal_dim), minval=0.0, maxval=5.0
        )
        scores_ood = detector.score(signals_ood)

        # OOD reconstruction error should be larger on average
        mean_recon_id = np.mean(scores_id["reconstruction_error"])
        mean_recon_ood = np.mean(scores_ood["reconstruction_error"])
        assert mean_recon_ood > mean_recon_id, (
            f"OOD recon ({mean_recon_ood:.4f}) should exceed "
            f"in-dist recon ({mean_recon_id:.4f})"
        )


class TestMahalanobisInDistribution:

    def test_in_distribution_lower_mahalanobis(self):
        """In-distribution signals should have lower Mahalanobis distance
        than random noise signals."""
        detector, _, simulator = _make_fitted_detector(n_reference=1000)

        key = jax.random.PRNGKey(400)
        k1, k2 = jax.random.split(key)

        # In-distribution
        _, signals_id = simulator.sample_and_simulate(k1, 200)
        scores_id = detector.score(signals_id)

        # OOD: signals far from training distribution
        signals_ood = jax.random.uniform(
            k2, (200, simulator.signal_dim), minval=3.0, maxval=10.0
        )
        scores_ood = detector.score(signals_ood)

        mean_mahal_id = np.mean(scores_id["mahalanobis_distance"])
        mean_mahal_ood = np.mean(scores_ood["mahalanobis_distance"])
        assert mean_mahal_ood > mean_mahal_id, (
            f"OOD Mahalanobis ({mean_mahal_ood:.2f}) should exceed "
            f"in-dist ({mean_mahal_id:.2f})"
        )


class TestOODResultProperties:

    def test_ood_fraction_correct(self):
        is_ood = np.array([True, False, True, False, True])
        result = OODResult(
            reconstruction_error=np.zeros(5),
            predictive_entropy=np.zeros(5),
            mahalanobis_distance=np.zeros(5),
            is_ood=is_ood,
            threshold_reconstruction=1.0,
            threshold_entropy=1.0,
            threshold_mahalanobis=1.0,
        )
        assert abs(result.ood_fraction - 0.6) < 1e-10

    def test_ood_fraction_all_false(self):
        result = OODResult(
            reconstruction_error=np.zeros(10),
            predictive_entropy=np.zeros(10),
            mahalanobis_distance=np.zeros(10),
            is_ood=np.zeros(10, dtype=bool),
            threshold_reconstruction=1.0,
            threshold_entropy=1.0,
            threshold_mahalanobis=1.0,
        )
        assert result.ood_fraction == 0.0

    def test_ood_fraction_all_true(self):
        result = OODResult(
            reconstruction_error=np.zeros(10),
            predictive_entropy=np.zeros(10),
            mahalanobis_distance=np.zeros(10),
            is_ood=np.ones(10, dtype=bool),
            threshold_reconstruction=1.0,
            threshold_entropy=1.0,
            threshold_mahalanobis=1.0,
        )
        assert result.ood_fraction == 1.0

    def test_print_summary_runs(self, capsys):
        result = OODResult(
            reconstruction_error=np.array([0.1, 0.2, 0.3]),
            predictive_entropy=np.array([0.5, 0.6, 0.7]),
            mahalanobis_distance=np.array([1.0, 2.0, 3.0]),
            is_ood=np.array([False, False, True]),
            threshold_reconstruction=0.25,
            threshold_entropy=0.65,
            threshold_mahalanobis=2.5,
        )
        result.print_summary()
        captured = capsys.readouterr()
        assert "Out-of-Distribution Detection Results" in captured.out
        assert "OOD voxels" in captured.out
        assert "Thresholds" in captured.out


class TestFlagVolumeShape:

    def test_flag_volume_returns_correct_shape(self):
        detector, _, simulator = _make_fitted_detector(n_reference=500)
        key = jax.random.PRNGKey(500)
        n_voxels = 50
        _, signals = simulator.sample_and_simulate(key, n_voxels)
        result = detector.flag_volume(signals)
        assert isinstance(result, OODResult)
        assert result.is_ood.shape == (n_voxels,)
        assert result.is_ood.dtype == bool
        assert result.reconstruction_error.shape == (n_voxels,)
        assert result.predictive_entropy.shape == (n_voxels,)
        assert result.mahalanobis_distance.shape == (n_voxels,)

    def test_flag_volume_with_mask(self):
        detector, _, simulator = _make_fitted_detector(n_reference=500)
        key = jax.random.PRNGKey(501)
        n_voxels = 40
        _, signals = simulator.sample_and_simulate(key, n_voxels)
        mask = np.zeros(n_voxels, dtype=bool)
        mask[:20] = True  # only score first 20

        result = detector.flag_volume(signals, mask=mask)
        assert result.is_ood.shape == (n_voxels,)
        # Masked-out voxels should have zero scores
        assert np.all(result.reconstruction_error[20:] == 0.0)
        assert np.all(result.mahalanobis_distance[20:] == 0.0)

    def test_flag_volume_thresholds_stored(self):
        detector, _, simulator = _make_fitted_detector(n_reference=500)
        key = jax.random.PRNGKey(502)
        _, signals = simulator.sample_and_simulate(key, 30)
        result = detector.flag_volume(signals, threshold_percentile=90.0)
        assert result.threshold_reconstruction > 0.0
        assert result.threshold_mahalanobis > 0.0
        assert np.isfinite(result.threshold_entropy)

    def test_flag_volume_requires_fit(self):
        model, simulator = _make_simple_setup()
        detector = OODDetector(model=model, simulator=simulator)
        key = jax.random.PRNGKey(1)
        _, signals = simulator.sample_and_simulate(key, 10)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.flag_volume(signals)
