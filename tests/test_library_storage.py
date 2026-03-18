"""
Tests for SimulationLibrary HDF5 I/O.
"""

import tempfile
import os

import jax.numpy as jnp
import numpy as np
import pytest

from dmipy_jax.library.storage import SimulationLibrary


class TestSimulationLibrary:

    def test_roundtrip_hdf5(self):
        """Save and load, verify data matches."""
        params = jnp.array(np.random.randn(500, 3).astype(np.float32))
        signals = jnp.array(np.random.randn(500, 64).astype(np.float32))
        lib = SimulationLibrary(
            params=params,
            signals=signals,
            parameter_names=["a", "b", "c"],
            metadata={"model": "test", "n_shells": 2},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_lib.h5")
            lib.save_hdf5(path)
            assert os.path.exists(path)

            loaded = SimulationLibrary.load_hdf5(path)
            assert loaded.n_entries == 500
            assert loaded.theta_dim == 3
            assert loaded.signal_dim == 64
            assert loaded.parameter_names == ["a", "b", "c"]
            assert jnp.allclose(loaded.params, params)
            assert jnp.allclose(loaded.signals, signals)

    def test_normalise_signals(self):
        signals = jnp.array([[3.0, 4.0], [1.0, 0.0]])
        lib = SimulationLibrary(
            params=jnp.zeros((2, 1)),
            signals=signals,
        )
        lib.normalise_signals()
        norms = jnp.linalg.norm(lib.signals, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_properties(self):
        lib = SimulationLibrary(
            params=jnp.zeros((10, 2)),
            signals=jnp.zeros((10, 32)),
        )
        assert lib.n_entries == 10
        assert lib.theta_dim == 2
        assert lib.signal_dim == 32
