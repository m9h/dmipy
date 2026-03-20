"""
ModelSimulator: unified wrapper that turns any forward model into a
(prior_sampler, simulate, add_noise) triple suitable for SBI training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, Dict, Optional, Tuple, Any
from jaxtyping import Array, Float, PRNGKeyArray


class ModelSimulator:
    """Wrap a forward model for use with the SBI pipeline.

    Parameters
    ----------
    forward_fn : callable
        ``(params_flat, acquisition) -> signal``  where *params_flat* is a 1-D
        array of length ``theta_dim`` and *signal* is ``(n_meas,)``.
    parameter_names : list of str
        Ordered names matching the flat parameter vector.
    parameter_ranges : dict
        ``{name: (low, high)}``.
    acquisition : object
        A ``JaxAcquisition`` (or duck-typed equivalent) passed as the second
        argument to *forward_fn*.
    noise_type : str
        ``"rician"`` (default) or ``"gaussian"``.
    snr : float
        Signal-to-noise ratio at b = 0.  ``sigma = 1 / snr``.
    constraints_fn : callable or None
        ``(key, theta_batch) -> theta_batch``  that enforces inter-parameter
        constraints (e.g. NODDI tortuosity).  Applied *after* prior sampling.
    prior_sampler_fn : callable or None
        Custom ``(key, n) -> (n, theta_dim)`` sampler.  If *None* the default
        uniform-within-ranges sampler is used.
    """

    def __init__(
        self,
        forward_fn: Callable,
        parameter_names: list,
        parameter_ranges: Dict[str, Tuple[float, float]],
        acquisition: Any,
        noise_type: str = "rician",
        snr: float = 30.0,
        constraints_fn: Optional[Callable] = None,
        prior_sampler_fn: Optional[Callable] = None,
        snr_range: Optional[Tuple[float, float]] = None,
    ):
        self.forward_fn = forward_fn
        self.parameter_names = list(parameter_names)
        self.parameter_ranges = dict(parameter_ranges)
        self.acquisition = acquisition
        self.noise_type = noise_type
        self.sigma = 1.0 / snr
        self.snr_range = snr_range
        self.constraints_fn = constraints_fn
        self._custom_prior = prior_sampler_fn
        self.curriculum_step: Optional[Tuple[int, int]] = None

        # Pre-compute flat bounds
        self._lows = jnp.array([self.parameter_ranges[n][0] for n in self.parameter_names])
        self._highs = jnp.array([self.parameter_ranges[n][1] for n in self.parameter_names])

    @property
    def theta_dim(self) -> int:
        return len(self.parameter_names)

    @property
    def signal_dim(self) -> int:
        """Number of measurements in the acquisition."""
        return int(self.acquisition.bvalues.shape[-1])

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def prior_sampler(self, key: PRNGKeyArray, n: int) -> Float[Array, "n theta_dim"]:
        """Sample *n* parameter vectors from the prior."""
        if self._custom_prior is not None:
            theta = self._custom_prior(key, n)
        else:
            theta = self._default_prior(key, n)

        if self.constraints_fn is not None:
            theta = self.constraints_fn(key, theta)
        return theta

    def simulate(self, key: PRNGKeyArray, theta: Float[Array, "n theta_dim"]) -> Float[Array, "n signal_dim"]:
        """Run the forward model on a batch of parameters (no noise)."""
        return jax.vmap(self._forward_single)(theta)

    def add_noise(self, key: PRNGKeyArray, signal: Float[Array, "n signal_dim"]) -> Float[Array, "n signal_dim"]:
        """Add noise to clean signal according to ``noise_type``."""
        if self.noise_type == "rician":
            return self._rician_noise(key, signal)
        else:
            return self._gaussian_noise(key, signal)

    def sample_and_simulate(
        self, key: PRNGKeyArray, n: int
    ) -> Tuple[Float[Array, "n theta_dim"], Float[Array, "n signal_dim"]]:
        """Convenience: sample prior, simulate, add noise.

        After adding noise the signals are normalised by their mean b0
        value (b-values < 100e6 s/m^2) so that training data matches the
        normalisation applied at inference time in ``SBIPredictor``.
        """
        k1, k2, k3 = random.split(key, 3)
        theta = self.prior_sampler(k1, n)
        signal = self.simulate(k2, theta)
        noisy = self.add_noise(k3, signal)

        # b0-normalise to match deploy-time preprocessing (deploy.py L93-99)
        b0_mask = self.acquisition.bvalues < 100e6
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean

        return theta, noisy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_single(self, params: Float[Array, "theta_dim"]) -> Float[Array, "signal_dim"]:
        return self.forward_fn(params, self.acquisition)

    def _default_prior(self, key: PRNGKeyArray, n: int) -> Float[Array, "n theta_dim"]:
        u = random.uniform(key, (n, self.theta_dim))
        return self._lows + u * (self._highs - self._lows)

    def _get_sigma(self, key: PRNGKeyArray, n: int):
        """Return per-sample sigma: shape ``(n, 1)`` when multi-SNR is active,
        scalar otherwise.  Safe inside ``jax.jit`` because branching is on
        plain Python attributes (traced as compile-time constants)."""
        if self.snr_range is not None:
            low_snr, high_snr = self.snr_range
            if self.curriculum_step is not None:
                step, total = self.curriculum_step
                progress = min(step / max(total, 1), 1.0)
                # Start with only high SNR, gradually include low SNR
                current_low = high_snr - progress * (high_snr - low_snr)
            else:
                current_low = low_snr
            snr_samples = random.uniform(key, (n, 1), minval=current_low, maxval=high_snr)
            return 1.0 / snr_samples
        return self.sigma

    def _rician_noise(self, key: PRNGKeyArray, signal: Float[Array, "n signal_dim"]) -> Float[Array, "n signal_dim"]:
        k1, k2, k3 = random.split(key, 3)
        sigma = self._get_sigma(k3, signal.shape[0])
        n1 = random.normal(k1, signal.shape) * sigma
        n2 = random.normal(k2, signal.shape) * sigma
        return jnp.sqrt((signal + n1) ** 2 + n2 ** 2)

    def _gaussian_noise(self, key: PRNGKeyArray, signal: Float[Array, "n signal_dim"]) -> Float[Array, "n signal_dim"]:
        k1, k2 = random.split(key)
        sigma = self._get_sigma(k2, signal.shape[0])
        noise = random.normal(k1, signal.shape) * sigma
        return signal + noise
