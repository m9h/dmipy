"""
DictionaryInitializer — use a pre-loaded simulation library for
initialising gradient-based fitting (same interface as GlobalBruteInitializer).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from dmipy_jax.library.storage import SimulationLibrary


class DictionaryInitializer:
    """Cosine-similarity best-match from a pre-loaded library.

    Drop-in replacement for :class:`GlobalBruteInitializer`.

    Parameters
    ----------
    library : SimulationLibrary
        Must already contain ``signals`` and ``params``.
    """

    def __init__(self, library: SimulationLibrary):
        self.library = library
        norms = jnp.linalg.norm(library.signals, axis=1, keepdims=True)
        norms = jnp.maximum(norms, 1e-12)
        self._lib_normed = library.signals / norms

    @partial(jax.jit, static_argnums=(0,))
    def compute_initial_guess(
        self,
        data: jnp.ndarray,
        acquisition=None,
        candidates=None,
    ) -> jnp.ndarray:
        """Return best-match parameters for a single voxel.

        Parameters
        ----------
        data : ``(M,)`` signal array.
        acquisition : ignored (kept for API compatibility).
        candidates : ignored.

        Returns
        -------
        params : ``(P,)``
        """
        s_norm = data / jnp.maximum(jnp.linalg.norm(data), 1e-12)
        cos_sim = self._lib_normed @ s_norm
        best_idx = jnp.argmax(cos_sim)
        return self.library.params[best_idx]

    @partial(jax.jit, static_argnums=(0,))
    def select_best_candidate(self, data, candidate_predictions=None, init_grid=None):
        """API-compatible alias for :meth:`compute_initial_guess`."""
        return self.compute_initial_guess(data)


class SBIInitializer:
    """Use a trained SBI model to provide initial parameter estimates.

    Parameters
    ----------
    model : eqx.Module
        Trained MDN or flow.
    """

    def __init__(self, model):
        self.model = model

    @partial(jax.jit, static_argnums=(0,))
    def compute_initial_guess(self, data, acquisition=None, candidates=None):
        """Return posterior-mean estimate for a single voxel signal."""
        import equinox as eqx
        from dmipy_jax.inference.mdn import MixtureDensityNetwork

        if isinstance(self.model, MixtureDensityNetwork):
            logits_pi, mu, log_sigma = self.model(data)
            weights = jax.nn.softmax(logits_pi)
            return jnp.sum(weights[:, None] * mu, axis=0)
        else:
            # Flow: sample and take mean
            key = jax.random.PRNGKey(0)
            samples = self.model.sample(key, condition=data, shape=(50,))
            return jnp.mean(samples, axis=0)
