"""
End-to-end pipeline test: train DTI MDN → save → load → predict on synthetic volume.
"""

import tempfile
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint
from dmipy_jax.inference.mdn import MixtureDensityNetwork


def _make_dti_simulator():
    """Build a simple DTI simulator that predicts FA and MD."""
    key = jax.random.PRNGKey(0)
    n_dirs = 32
    bvecs = jax.random.normal(key, (n_dirs, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.full(n_dirs, 1e9)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        """Simplified DTI: params = [FA_scaled, MD_scaled] where MD_scaled = MD * 1e9."""
        fa = jnp.clip(params[0], 0.0, 1.0)
        md = jnp.clip(params[1], 0.1, 3.0) * 1e-9  # un-scale

        # Reconstruct eigenvalues from FA and MD
        # Simple model: l1 = MD + 2/3 * FA * MD, l2=l3 = MD - 1/3 * FA * MD
        l1 = md * (1 + 2.0 / 3.0 * fa)
        l23 = md * (1 - 1.0 / 3.0 * fa)

        # Signal = mean over directions of exp(-b * (l1*cos^2 + l23*sin^2))
        # For simplicity, use isotropic average
        d_avg = (l1 + 2 * l23) / 3.0
        return jnp.exp(-acq.bvalues * d_avg)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["FA", "MD_scaled"],
        parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
        acquisition=acq,
        noise_type="rician",
        snr=30.0,
    ), acq


class TestEndToEnd:

    @pytest.mark.slow
    def test_train_save_load_predict(self):
        """Train 100 steps → save → load → predict on small synthetic volume."""
        sim, acq = _make_dti_simulator()

        config = SBIPipelineConfig(
            model_name="DTI_test",
            parameter_names=["FA", "MD_scaled"],
            parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
            acquisition={"bvalues": acq.bvalues.tolist()},
            inference_mode="mdn",
            n_components=4,
            hidden_dim=64,
            depth=2,
            learning_rate=1e-3,
            batch_size=128,
            n_steps=100,
            snr=30.0,
        )

        # Train
        key = jax.random.PRNGKey(123)
        model, losses = train_sbi(config, sim, key=key, print_every=50)
        assert len(losses) == 100
        assert losses[-1] < losses[0], "Loss should decrease"

        # Save + load round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dti_e2e")
            save_checkpoint(model, config, path)
            loaded_model, loaded_config = load_checkpoint(path)

            # Predict on synthetic 10x10x10 volume
            vol_shape = (10, 10, 10)
            n_vox = np.prod(vol_shape)
            key_test = jax.random.PRNGKey(999)
            theta_test, signals_test = sim.sample_and_simulate(key_test, n_vox)

            # Batch predict
            @jax.jit
            def predict_mean(x):
                logits_pi, mu, log_sigma = loaded_model(x)
                weights = jax.nn.softmax(logits_pi)
                return jnp.sum(weights[:, None] * mu, axis=0)

            preds = jax.vmap(predict_mean)(signals_test)

            # Verify FA is in [0, 1] range (approximately)
            fa_pred = np.asarray(preds[:, 0])
            assert np.all(fa_pred > -0.5), f"FA min: {fa_pred.min()}"
            assert np.all(fa_pred < 1.5), f"FA max: {fa_pred.max()}"

            # Verify MD > 0 (approximately)
            md_pred = np.asarray(preds[:, 1])
            assert np.all(md_pred > -1.0), f"MD min: {md_pred.min()}"
