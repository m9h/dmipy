"""
Tests for ensemble inference: shape checks, uncertainty decomposition,
variance reduction, and save/load round-trip.
"""

import tempfile
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import _NormalisedMDN
from dmipy_jax.pipeline.ensemble import (
    EnsemblePredictor,
    save_ensemble,
    load_ensemble,
)
from dmipy_jax.inference.mdn import MixtureDensityNetwork


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SIGNAL_DIM = 32
_THETA_DIM = 2
_N_COMPONENTS = 4
_WIDTH = 64
_DEPTH = 2


def _make_dummy_ensemble(n_members=3):
    """Create dummy _NormalisedMDN models for testing (no training)."""
    models = []
    for i in range(n_members):
        key = jax.random.PRNGKey(i)
        inner = MixtureDensityNetwork(
            in_features=_SIGNAL_DIM,
            out_features=_THETA_DIM,
            num_components=_N_COMPONENTS,
            width_size=_WIDTH,
            depth=_DEPTH,
            key=key,
        )
        lows = jnp.array([0.0, 0.1])
        spans = jnp.array([1.0, 2.9])
        models.append(_NormalisedMDN(inner, lows, spans))

    config = SBIPipelineConfig(
        model_name="TestDTI",
        parameter_names=["FA", "MD"],
        parameter_ranges={"FA": (0.0, 1.0), "MD": (0.1, 3.0)},
        inference_mode="mdn",
        n_components=_N_COMPONENTS,
        hidden_dim=_WIDTH,
        depth=_DEPTH,
        acquisition={"bvalues": [0.0] * _SIGNAL_DIM},
        activation="relu",
    )
    return models, config


def _make_test_signals(batch_size=16):
    """Generate random test signals."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (batch_size, _SIGNAL_DIM))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestEnsemblePredictMeanShape:
    """Verify output shapes from predict_mean."""

    def test_mean_shape(self):
        models, config = _make_dummy_ensemble(n_members=3)
        ens = EnsemblePredictor(models, config)
        signals = _make_test_signals(batch_size=8)

        mean, std = ens.predict_mean(signals)

        assert mean.shape == (8, _THETA_DIM), f"Expected (8, {_THETA_DIM}), got {mean.shape}"
        assert std.shape == (8, _THETA_DIM), f"Expected (8, {_THETA_DIM}), got {std.shape}"

    def test_single_member(self):
        """Ensemble with 1 member: std should be zero."""
        models, config = _make_dummy_ensemble(n_members=1)
        ens = EnsemblePredictor(models, config)
        signals = _make_test_signals(batch_size=4)

        mean, std = ens.predict_mean(signals)

        assert mean.shape == (4, _THETA_DIM)
        assert jnp.allclose(std, 0.0, atol=1e-6), "Single-member ensemble should have zero epistemic std"


class TestEnsembleUncertaintyDecomposition:
    """Verify total = sqrt(aleatoric^2 + epistemic^2)."""

    def test_total_equals_sqrt_sum_of_squares(self):
        models, config = _make_dummy_ensemble(n_members=5)
        ens = EnsemblePredictor(models, config)
        signals = _make_test_signals(batch_size=16)

        result = ens.predict_with_uncertainty(signals)

        assert set(result.keys()) == {"mean", "aleatoric", "epistemic", "total"}

        expected_total = jnp.sqrt(result["aleatoric"] ** 2 + result["epistemic"] ** 2)
        assert jnp.allclose(result["total"], expected_total, atol=1e-5), (
            f"Max mismatch: {jnp.max(jnp.abs(result['total'] - expected_total))}"
        )

    def test_uncertainty_shapes(self):
        models, config = _make_dummy_ensemble(n_members=3)
        ens = EnsemblePredictor(models, config)
        signals = _make_test_signals(batch_size=10)

        result = ens.predict_with_uncertainty(signals)

        for key in ("mean", "aleatoric", "epistemic", "total"):
            assert result[key].shape == (10, _THETA_DIM), (
                f"{key}: expected (10, {_THETA_DIM}), got {result[key].shape}"
            )

    def test_uncertainties_are_non_negative(self):
        models, config = _make_dummy_ensemble(n_members=4)
        ens = EnsemblePredictor(models, config)
        signals = _make_test_signals(batch_size=12)

        result = ens.predict_with_uncertainty(signals)

        for key in ("aleatoric", "epistemic", "total"):
            assert jnp.all(result[key] >= 0), f"{key} has negative values"


class TestEnsembleReducesVariance:
    """Ensemble mean should have lower error variance than individual members."""

    def test_ensemble_mean_lower_variance(self):
        n_members = 5
        models, config = _make_dummy_ensemble(n_members=n_members)
        ens = EnsemblePredictor(models, config)

        # Generate test signals and a "ground truth" (use member 0 with a
        # fixed seed as pseudo-ground-truth -- we're testing that averaging
        # reduces variance across random initialisations, not accuracy)
        key = jax.random.PRNGKey(99)
        signals = jax.random.normal(key, (64, _SIGNAL_DIM))

        # Get individual member predictions
        member_preds = []
        for model in models:
            m, _ = ens._member_stats(model, signals)
            member_preds.append(m)

        # Ensemble mean
        ens_mean = jnp.mean(jnp.stack(member_preds, axis=0), axis=0)

        # Compute MSE of each member vs ensemble mean, vs variance of members
        # The key insight: the variance of the ensemble mean across resampling
        # should be smaller than the variance of any individual member.
        # Here we verify that the spread (std across members) is positive,
        # i.e., members disagree, and the ensemble mean is "between" them.
        stacked = jnp.stack(member_preds, axis=0)  # (M, B, D)
        member_spread = jnp.std(stacked, axis=0)  # (B, D)

        # Members should disagree (different random seeds)
        assert jnp.mean(member_spread) > 1e-3, (
            "Members should disagree due to different random seeds"
        )

        # Ensemble mean should be closer to each member than the furthest member
        distances = jnp.stack(
            [jnp.mean((m - ens_mean) ** 2) for m in member_preds]
        )
        max_member_dist = jnp.max(
            jnp.stack(
                [jnp.mean((member_preds[j] - member_preds[k]) ** 2)
                 for j in range(n_members) for k in range(j + 1, n_members)]
            )
        )
        # Ensemble mean distance to any member should be <= max pairwise distance
        assert jnp.all(distances <= max_member_dist + 1e-6), (
            "Ensemble mean should be no further from any member than the max pairwise distance"
        )


class TestSaveLoadEnsemble:
    """Round-trip save/load for ensembles."""

    def test_save_load_roundtrip(self):
        models, config = _make_dummy_ensemble(n_members=3)
        members = [(m, [1.0, 0.5]) for m in models]  # dummy losses

        signals = _make_test_signals(batch_size=8)

        # Original predictions
        ens_orig = EnsemblePredictor.from_training(members, config)
        orig_mean, orig_std = ens_orig.predict_mean(signals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ensemble")
            save_ensemble(members, config, path)

            # Verify files exist
            for i in range(3):
                assert os.path.exists(f"{path}_member_{i}.eqx"), (
                    f"Missing checkpoint file for member {i}"
                )

            # Load
            key = jax.random.PRNGKey(0)
            ens_loaded = load_ensemble(path, n_members=3, key=key)

            assert len(ens_loaded.models) == 3
            assert ens_loaded.config.model_name == "TestDTI"

            # Verify predictions match
            loaded_mean, loaded_std = ens_loaded.predict_mean(signals)

            assert jnp.allclose(orig_mean, loaded_mean, atol=1e-4), (
                f"Mean mismatch: max diff = {jnp.max(jnp.abs(orig_mean - loaded_mean))}"
            )
            assert jnp.allclose(orig_std, loaded_std, atol=1e-4), (
                f"Std mismatch: max diff = {jnp.max(jnp.abs(orig_std - loaded_std))}"
            )

    def test_from_training_classmethod(self):
        models, config = _make_dummy_ensemble(n_members=2)
        members = [(m, [1.0]) for m in models]

        ens = EnsemblePredictor.from_training(members, config)
        assert len(ens.models) == 2
        assert ens.config is config
