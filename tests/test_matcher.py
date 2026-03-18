"""
Tests for DictionaryMatcher — cosine-similarity matching.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.library.matcher import DictionaryMatcher


def _build_ball_library():
    """Build a small Ball-model library for testing."""
    key = jax.random.PRNGKey(0)
    n_meas = 32
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.full(n_meas, 1e9)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        return jnp.exp(-acq.bvalues * params[0])

    sim = ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d"],
        parameter_ranges={"d": (0.1e-9, 3.0e-9)},
        acquisition=acq,
    )

    gen = LibraryGenerator(sim, chunk_size=5000)
    params, signals = gen.generate(5000, key=jax.random.PRNGKey(1))

    lib = SimulationLibrary(
        params=params,
        signals=signals,
        parameter_names=["d"],
    )
    return lib, sim


class TestDictionaryMatcher:

    def test_match_single_recovers_known(self):
        """A signal generated from a known parameter should match that parameter."""
        lib, sim = _build_ball_library()
        matcher = DictionaryMatcher(lib, k_best=5)

        # Pick a known library entry
        idx = 100
        true_params = lib.params[idx]
        signal = lib.signals[idx]

        best, topk = matcher.match_single(signal)
        # Best match should be exact (or very close) since it's in the library
        assert jnp.allclose(best, true_params, atol=1e-5)

    def test_match_batch(self):
        lib, sim = _build_ball_library()
        matcher = DictionaryMatcher(lib, k_best=3)

        signals = lib.signals[:10]
        best, topk = matcher.match_batch(signals)
        assert best.shape == (10, 1)
        assert topk.shape == (10, 3, 1)

    def test_topk_iqr(self):
        lib, sim = _build_ball_library()
        matcher = DictionaryMatcher(lib, k_best=5)

        signals = lib.signals[:5]
        _, topk = matcher.match_batch(signals)
        iqr = DictionaryMatcher.topk_iqr(topk)
        assert iqr.shape == (5, 1)
        assert jnp.all(iqr >= 0)

    def test_match_noisy_signal(self):
        """A noisy version of a library signal should still match approximately."""
        lib, sim = _build_ball_library()
        matcher = DictionaryMatcher(lib, k_best=5)

        idx = 200
        true_d = lib.params[idx, 0]
        signal = lib.signals[idx]

        # Add moderate noise
        key = jax.random.PRNGKey(42)
        noisy = signal + jax.random.normal(key, signal.shape) * 0.02

        best, _ = matcher.match_single(noisy)
        # Should be within 50% of true value (Ball model has limited resolution)
        assert jnp.abs(best[0] - true_d) / true_d < 0.5
