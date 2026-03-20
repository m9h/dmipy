"""
Tests for checkpoint save/load round-trip.
"""

import tempfile
import os

import jax
import jax.numpy as jnp
import pytest

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint
from dmipy_jax.pipeline.train import _NormalisedMDN
from dmipy_jax.inference.mdn import MixtureDensityNetwork


class TestCheckpointRoundTrip:

    def test_mdn_roundtrip(self):
        """Save and load a _NormalisedMDN, verify predictions match."""
        key = jax.random.PRNGKey(42)

        # Build a _NormalisedMDN (matches what train_sbi returns)
        inner = MixtureDensityNetwork(
            in_features=32,
            out_features=2,
            num_components=4,
            width_size=64,
            depth=2,
            key=key,
        )
        lows = jnp.array([0.0, 0.1])
        spans = jnp.array([1.0, 2.9])
        model = _NormalisedMDN(inner, lows, spans)

        config = SBIPipelineConfig(
            model_name="TestDTI",
            parameter_names=["FA", "MD"],
            inference_mode="mdn",
            n_components=4,
            hidden_dim=64,
            depth=2,
            acquisition={"bvalues": [0.0] * 32},
        )

        # Generate a test input
        x = jax.random.normal(key, (5, 32))
        original_out = jax.vmap(model)(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ckpt")
            save_checkpoint(model, config, path)

            # Verify files exist
            assert os.path.exists(path + ".eqx")
            assert os.path.exists(os.path.join(tmpdir, "test_ckpt.config.json"))

            # Load
            loaded_model, loaded_config = load_checkpoint(path, key=key)

            # Verify config
            assert loaded_config.model_name == "TestDTI"
            assert loaded_config.parameter_names == ["FA", "MD"]

            # Verify it's a _NormalisedMDN
            assert isinstance(loaded_model, _NormalisedMDN)

            # Verify predictions match
            loaded_out = jax.vmap(loaded_model)(x)
            for orig, loaded in zip(original_out, loaded_out):
                assert jnp.allclose(orig, loaded, atol=1e-5), \
                    f"Mismatch: max diff = {jnp.max(jnp.abs(orig - loaded))}"

    def test_config_serialization(self):
        """Config round-trips through JSON."""
        config = SBIPipelineConfig(
            model_name="NODDI",
            parameter_names=["f_intra", "f_iso", "kappa"],
            snr=50.0,
            n_steps=1000,
        )
        d = config.to_dict()
        restored = SBIPipelineConfig.from_dict(d)
        assert restored.model_name == "NODDI"
        assert restored.snr == 50.0
        assert restored.theta_dim == 3
