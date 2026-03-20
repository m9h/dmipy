"""Tests for MDN architectures -- MLP vs Residual."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from dmipy_jax.inference.mdn import (
    MixtureDensityNetwork,
    ResidualMDN,
    mdn_loss,
)


# Shared test parameters
IN_FEATURES = 32
OUT_FEATURES = 3
NUM_COMPONENTS = 4
WIDTH_SIZE = 64
DEPTH = 3


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mlp_model(key):
    return MixtureDensityNetwork(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        num_components=NUM_COMPONENTS,
        width_size=WIDTH_SIZE,
        depth=DEPTH,
        key=key,
    )


@pytest.fixture
def residual_model(key):
    return ResidualMDN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        num_components=NUM_COMPONENTS,
        width_size=WIDTH_SIZE,
        depth=DEPTH,
        key=key,
    )


class TestResidualMDN:

    def test_residual_mdn_output_shape(self, residual_model, mlp_model, key):
        """Verify (logits_pi, mu, log_sigma) shapes match MLP MDN."""
        x = jax.random.normal(key, (IN_FEATURES,))

        mlp_logits, mlp_mu, mlp_log_sigma = mlp_model(x)
        res_logits, res_mu, res_log_sigma = residual_model(x)

        assert res_logits.shape == mlp_logits.shape == (NUM_COMPONENTS,)
        assert res_mu.shape == mlp_mu.shape == (NUM_COMPONENTS, OUT_FEATURES)
        assert res_log_sigma.shape == mlp_log_sigma.shape == (NUM_COMPONENTS, OUT_FEATURES)

    def test_residual_mdn_forward_pass(self, residual_model, key):
        """Verify forward pass runs without error."""
        x = jax.random.normal(key, (IN_FEATURES,))
        logits_pi, mu, log_sigma = residual_model(x)

        # Outputs should be finite
        assert jnp.all(jnp.isfinite(logits_pi))
        assert jnp.all(jnp.isfinite(mu))
        assert jnp.all(jnp.isfinite(log_sigma))

    def test_residual_mdn_loss_computes(self, residual_model, key):
        """Verify mdn_loss works with ResidualMDN."""
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (IN_FEATURES,))
        y = jax.random.normal(k2, (OUT_FEATURES,))

        loss = mdn_loss(residual_model, x, y)

        assert jnp.isfinite(loss)
        assert loss.shape == ()

    def test_residual_mdn_gradients_flow(self, residual_model, key):
        """Verify eqx.filter_grad works (no NaN grads)."""
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (IN_FEATURES,))
        y = jax.random.normal(k2, (OUT_FEATURES,))

        grads = eqx.filter_grad(mdn_loss)(residual_model, x, y)

        # Check that all gradient leaves are finite (no NaN, no Inf)
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0, "Expected at least one gradient array"
        for leaf in grad_leaves:
            assert jnp.all(jnp.isfinite(leaf)), (
                f"Non-finite gradient found with shape {leaf.shape}"
            )

    def test_residual_matches_mlp_interface(self, residual_model, mlp_model, key):
        """Both architectures return the same tuple structure."""
        x = jax.random.normal(key, (IN_FEATURES,))

        mlp_out = mlp_model(x)
        res_out = residual_model(x)

        # Both return 3-tuples
        assert len(mlp_out) == 3
        assert len(res_out) == 3

        # Each element has the same dtype
        for m, r in zip(mlp_out, res_out):
            assert m.dtype == r.dtype
