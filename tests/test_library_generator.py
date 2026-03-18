"""
Tests for LibraryGenerator — shape verification and consistency.
"""

import jax
import jax.numpy as jnp
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator


def _make_ball_stick_simulator():
    """Simulator for Ball+Stick (2 params: d_ball, d_stick_par)."""
    key = jax.random.PRNGKey(0)
    n_meas = 32
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.full(n_meas, 1e9)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        d_ball = params[0]
        d_stick = params[1]
        f = 0.5  # fixed fraction
        s_ball = jnp.exp(-acq.bvalues * d_ball)
        # Stick along z
        cos2 = acq.gradient_directions[:, 2] ** 2
        s_stick = jnp.exp(-acq.bvalues * d_stick * cos2)
        return f * s_ball + (1 - f) * s_stick

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_ball", "d_stick"],
        parameter_ranges={"d_ball": (0.5e-9, 3.0e-9), "d_stick": (0.5e-9, 3.0e-9)},
        acquisition=acq,
    )


class TestLibraryGenerator:

    def test_generate_shapes(self):
        sim = _make_ball_stick_simulator()
        gen = LibraryGenerator(sim, chunk_size=5000)
        params, signals = gen.generate(10000)
        assert params.shape == (10000, 2)
        assert signals.shape == (10000, 32)

    def test_generate_small(self):
        sim = _make_ball_stick_simulator()
        gen = LibraryGenerator(sim, chunk_size=500)
        params, signals = gen.generate(100)
        assert params.shape == (100, 2)
        assert signals.shape == (100, 32)

    def test_params_in_range(self):
        sim = _make_ball_stick_simulator()
        gen = LibraryGenerator(sim, chunk_size=5000)
        params, _ = gen.generate(1000)
        assert jnp.all(params[:, 0] >= 0.5e-9)
        assert jnp.all(params[:, 0] <= 3.0e-9)
        assert jnp.all(params[:, 1] >= 0.5e-9)
        assert jnp.all(params[:, 1] <= 3.0e-9)

    def test_signals_positive(self):
        sim = _make_ball_stick_simulator()
        gen = LibraryGenerator(sim, chunk_size=5000)
        _, signals = gen.generate(1000)
        assert jnp.all(signals >= 0)
        assert jnp.all(signals <= 1.0 + 1e-6)

    def test_spot_check_vs_individual(self):
        """Spot-check 10 entries against individual model calls."""
        sim = _make_ball_stick_simulator()
        gen = LibraryGenerator(sim, chunk_size=5000)
        key = jax.random.PRNGKey(42)
        params, signals = gen.generate(100, key=key)

        for i in range(10):
            expected = sim._forward_single(params[i])
            assert jnp.allclose(signals[i], expected, atol=1e-5), \
                f"Entry {i} mismatch: max diff={jnp.max(jnp.abs(signals[i] - expected))}"
