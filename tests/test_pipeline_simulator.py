"""
Tests for ModelSimulator — shapes, noise stats, constraint callback.
"""

import jax
import jax.numpy as jnp
import pytest

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_ball_simulator(snr=30.0, noise_type="rician"):
    """Simple isotropic Ball model simulator for testing."""
    n_meas = 32
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    bvals = jnp.full(n_meas, 1e9)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

    def forward_fn(params, acq):
        d = params[0]
        return jnp.exp(-acq.bvalues * d)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["lambda_iso"],
        parameter_ranges={"lambda_iso": (0.1e-9, 3.0e-9)},
        acquisition=acq,
        noise_type=noise_type,
        snr=snr,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestModelSimulatorShapes:

    def test_prior_shape(self):
        sim = _make_ball_simulator()
        key = jax.random.PRNGKey(1)
        theta = sim.prior_sampler(key, 100)
        assert theta.shape == (100, 1)

    def test_simulate_shape(self):
        sim = _make_ball_simulator()
        key = jax.random.PRNGKey(2)
        theta = sim.prior_sampler(key, 50)
        sig = sim.simulate(key, theta)
        assert sig.shape == (50, 32)

    def test_add_noise_shape(self):
        sim = _make_ball_simulator()
        key = jax.random.PRNGKey(3)
        sig = jnp.ones((20, 32))
        noisy = sim.add_noise(key, sig)
        assert noisy.shape == (20, 32)

    def test_sample_and_simulate(self):
        sim = _make_ball_simulator()
        key = jax.random.PRNGKey(4)
        theta, sig = sim.sample_and_simulate(key, 64)
        assert theta.shape == (64, 1)
        assert sig.shape == (64, 32)


class TestNoiseStatistics:

    def test_rician_noise_positive(self):
        sim = _make_ball_simulator(snr=30.0, noise_type="rician")
        key = jax.random.PRNGKey(5)
        sig = jnp.ones((10000, 32)) * 0.5
        noisy = sim.add_noise(key, sig)
        # Rician noise is always positive
        assert jnp.all(noisy >= 0)

    def test_rician_noise_mean_bias(self):
        """Rician noise should bias the mean upward relative to clean signal."""
        sim = _make_ball_simulator(snr=10.0, noise_type="rician")
        key = jax.random.PRNGKey(6)
        clean = jnp.ones((50000, 32)) * 0.3
        noisy = sim.add_noise(key, clean)
        # Mean of Rician-corrupted signal should be >= mean of clean
        assert jnp.mean(noisy) >= jnp.mean(clean) - 0.01

    def test_gaussian_noise_mean(self):
        """Gaussian noise should preserve mean approximately."""
        sim = _make_ball_simulator(snr=30.0, noise_type="gaussian")
        key = jax.random.PRNGKey(7)
        clean = jnp.ones((50000, 32)) * 0.5
        noisy = sim.add_noise(key, clean)
        assert jnp.abs(jnp.mean(noisy) - 0.5) < 0.01


class TestConstraintCallback:

    def test_tortuosity_constraint(self):
        """Verify NODDI tortuosity: d_perp = d_par * (1 - f_intra)."""
        n_meas = 32
        key = jax.random.PRNGKey(0)
        bvecs = jax.random.normal(key, (n_meas, 3))
        bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
        bvals = jnp.full(n_meas, 1e9)
        acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)

        d_par = 1.7e-9

        def forward_fn(params, acq):
            return jnp.exp(-acq.bvalues * params[0])

        def tortuosity(key, theta):
            # theta[:, 0] = f_intra, theta[:, 1] = d_perp (to be constrained)
            f_intra = theta[:, 0]
            d_perp = d_par * (1.0 - f_intra)
            return theta.at[:, 1].set(d_perp)

        sim = ModelSimulator(
            forward_fn=forward_fn,
            parameter_names=["f_intra", "d_perp"],
            parameter_ranges={"f_intra": (0.01, 0.99), "d_perp": (0.0, 3e-9)},
            acquisition=acq,
            constraints_fn=tortuosity,
        )

        key = jax.random.PRNGKey(10)
        theta = sim.prior_sampler(key, 1000)
        expected_d_perp = d_par * (1.0 - theta[:, 0])
        assert jnp.allclose(theta[:, 1], expected_d_perp, atol=1e-15)
