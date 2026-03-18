"""
Tests for Bingham-NODDI SBI pipeline — training, posterior recovery,
and kappa asymmetry inference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.models.bingham_noddi import (
    build_bingham_noddi_simulator,
    get_bingham_noddi_config,
)


def _make_acq():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (2, 1))
    v1 = rand_vecs(k1, 16)
    v2 = rand_vecs(k2, 16)
    bvals = jnp.concatenate([jnp.zeros(2), jnp.full(16, 700e6), jnp.full(16, 2000e6)])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


class TestBinghamSimulator:

    def test_simulator_shapes(self):
        acq = _make_acq()
        sim = build_bingham_noddi_simulator(acq, snr=30.0)
        key = jax.random.PRNGKey(1)
        theta, signals = sim.sample_and_simulate(key, 50)
        assert theta.shape == (50, 5)
        assert signals.shape == (50, 34)

    def test_signals_positive(self):
        acq = _make_acq()
        sim = build_bingham_noddi_simulator(acq, snr=30.0)
        key = jax.random.PRNGKey(2)
        _, signals = sim.sample_and_simulate(key, 100)
        assert jnp.all(signals > 0)

    def test_config_roundtrip(self):
        acq = _make_acq()
        config = get_bingham_noddi_config(acq)
        assert config.model_name == "BinghamNODDI"
        assert config.theta_dim == 5
        d = config.to_dict()
        restored = SBIPipelineConfig.from_dict(d)
        assert restored.model_name == "BinghamNODDI"


class TestBinghamSBITraining:

    @pytest.mark.slow
    def test_training_loss_decreases(self):
        """Train 200 steps and verify loss decreases."""
        acq = _make_acq()
        sim = build_bingham_noddi_simulator(acq, snr=30.0)
        config = get_bingham_noddi_config(acq, n_steps=200)
        config.n_steps = 200
        config.hidden_dim = 64
        config.depth = 2
        config.n_components = 4
        config.batch_size = 128

        model, losses = train_sbi(config, sim, print_every=0)
        assert len(losses) == 200
        # Loss should decrease overall (compare first 10 avg to last 10 avg)
        early = np.mean(losses[:10])
        late = np.mean(losses[-10:])
        assert late < early, f"Loss did not decrease: {early:.4f} -> {late:.4f}"

    @pytest.mark.slow
    def test_posterior_recovery(self):
        """Train 500 steps and check parameter recovery on test set."""
        acq = _make_acq()
        sim = build_bingham_noddi_simulator(acq, snr=30.0)
        config = get_bingham_noddi_config(acq, n_steps=500)
        config.n_steps = 500
        config.hidden_dim = 64
        config.depth = 2
        config.n_components = 4
        config.batch_size = 128

        model, losses = train_sbi(config, sim, print_every=0)

        # Test set
        key = jax.random.PRNGKey(42)
        theta_test, signals_test = sim.sample_and_simulate(key, 200)

        @eqx.filter_jit
        def predict_mean(mdl, x):
            logits_pi, mu, log_sigma = mdl(x)
            weights = jax.nn.softmax(logits_pi)
            return jnp.sum(weights[:, None] * mu, axis=0)

        preds = np.asarray(jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_test))
        theta_np = np.asarray(theta_test)

        # f_intra (index 0) should have some correlation > 0
        r_fintra = np.corrcoef(theta_np[:, 0], preds[:, 0])[0, 1]
        assert r_fintra > 0.0, f"f_intra correlation too low: {r_fintra:.4f}"

    @pytest.mark.slow
    def test_kappa_asymmetry_recoverable(self):
        """Verify the model can distinguish kappa1 != kappa2."""
        acq = _make_acq()
        sim = build_bingham_noddi_simulator(acq, snr=30.0)
        config = get_bingham_noddi_config(acq, n_steps=500)
        config.n_steps = 500
        config.hidden_dim = 64
        config.depth = 2
        config.n_components = 4
        config.batch_size = 128

        model, _ = train_sbi(config, sim, print_every=0)

        key = jax.random.PRNGKey(99)
        theta_test, signals_test = sim.sample_and_simulate(key, 500)

        @eqx.filter_jit
        def predict_mean(mdl, x):
            logits_pi, mu, log_sigma = mdl(x)
            weights = jax.nn.softmax(logits_pi)
            return jnp.sum(weights[:, None] * mu, axis=0)

        preds = np.asarray(jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_test))
        theta_np = np.asarray(theta_test)

        # kappa1 and kappa2 are indices 2 and 3
        # Check they are NOT perfectly correlated with each other in predictions
        # (i.e. the model doesn't just predict k1 == k2 always)
        k1_pred = preds[:, 2]
        k2_pred = preds[:, 3]
        # If model always predicts k1==k2, std of (k1-k2) would be ~0
        diff_std = np.std(k1_pred - k2_pred)
        assert diff_std > 0.01, f"Model predicts k1≈k2 always (diff std={diff_std:.4f})"
