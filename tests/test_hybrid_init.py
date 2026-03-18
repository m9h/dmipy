"""
Tests for DictionaryInitializer and SBIInitializer.
"""

import jax
import jax.numpy as jnp
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.fitting.hybrid_initializer import DictionaryInitializer


def _build_ball_library():
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
    lib = SimulationLibrary(params=params, signals=signals, parameter_names=["d"])
    return lib, sim


class TestDictionaryInitializer:

    def test_returns_correct_shape(self):
        lib, _ = _build_ball_library()
        init = DictionaryInitializer(lib)
        signal = lib.signals[50]
        result = init.compute_initial_guess(signal)
        assert result.shape == (1,)

    def test_recovers_known_entry(self):
        lib, _ = _build_ball_library()
        init = DictionaryInitializer(lib)
        idx = 300
        signal = lib.signals[idx]
        result = init.compute_initial_guess(signal)
        assert jnp.allclose(result, lib.params[idx], atol=1e-5)

    def test_api_compatibility(self):
        """select_best_candidate should work as alias."""
        lib, _ = _build_ball_library()
        init = DictionaryInitializer(lib)
        signal = lib.signals[10]
        r1 = init.compute_initial_guess(signal)
        r2 = init.select_best_candidate(signal)
        assert jnp.allclose(r1, r2)
