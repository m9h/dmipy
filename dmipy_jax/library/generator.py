"""
LibraryGenerator: batch-generate (params, signals) libraries via ``jax.vmap``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from dmipy_jax.pipeline.simulator import ModelSimulator


class LibraryGenerator:
    """Generate large signal dictionaries from a :class:`ModelSimulator`.

    Parameters
    ----------
    simulator : ModelSimulator
        Simulator wrapping the forward model + prior.
    chunk_size : int
        Number of entries to simulate per GPU call (controls memory).
    """

    def __init__(self, simulator: ModelSimulator, chunk_size: int = 50_000):
        self.simulator = simulator
        self.chunk_size = chunk_size

    def generate(
        self,
        n_entries: int,
        key: Optional[PRNGKeyArray] = None,
        add_noise: bool = False,
    ) -> Tuple[Float[Array, "n theta_dim"], Float[Array, "n signal_dim"]]:
        """Generate a library of parameter–signal pairs.

        Parameters
        ----------
        n_entries : int
            Total number of library entries.
        key : jax.Array, optional
            PRNG key.  Defaults to ``PRNGKey(0)``.
        add_noise : bool
            Whether to corrupt signals with the simulator's noise model.

        Returns
        -------
        params : jnp.ndarray  ``(n_entries, theta_dim)``
        signals : jnp.ndarray ``(n_entries, signal_dim)``
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        n_chunks = math.ceil(n_entries / self.chunk_size)
        all_params = []
        all_signals = []

        for i in range(n_chunks):
            key, k_sample, k_sim, k_noise = jax.random.split(key, 4)
            size = min(self.chunk_size, n_entries - i * self.chunk_size)

            theta = self.simulator.prior_sampler(k_sample, size)
            sig = self.simulator.simulate(k_sim, theta)

            if add_noise:
                sig = self.simulator.add_noise(k_noise, sig)

            all_params.append(theta)
            all_signals.append(sig)

        params = jnp.concatenate(all_params, axis=0)
        signals = jnp.concatenate(all_signals, axis=0)
        return params, signals
