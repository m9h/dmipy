"""
Dictionary matching via cosine similarity — pure JAX, no FAISS.

For a library of *N* entries and *V* voxels each with *M* measurements,
matching is a single matmul ``(V, M) @ (N, M).T`` batched via ``jax.vmap``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dmipy_jax.library.storage import SimulationLibrary


class DictionaryMatcher:
    """FORCE-style dictionary matching using cosine similarity.

    Parameters
    ----------
    library : SimulationLibrary
        Pre-computed signal library (will be L2-normalised internally).
    k_best : int
        Number of top-K matches to retain for uncertainty estimation.
    alpha : float
        Penalty weight for fiber count in penalised matching.
        ``score = cos(S, S_i) - alpha * n_fibers(i)``.
    """

    def __init__(
        self,
        library: SimulationLibrary,
        k_best: int = 5,
        alpha: float = 0.0,
    ):
        self.library = library
        self.k_best = k_best
        self.alpha = alpha

        # L2-normalise library signals for cosine similarity
        norms = jnp.linalg.norm(library.signals, axis=1, keepdims=True)
        norms = jnp.maximum(norms, 1e-12)
        self._lib_signals_normed = library.signals / norms

    # ------------------------------------------------------------------ #
    # Core matching
    # ------------------------------------------------------------------ #

    def match_single(
        self,
        signal: Float[Array, "M"],
        penalty: Optional[Float[Array, "N"]] = None,
    ) -> Tuple[Float[Array, "theta_dim"], Float[Array, "k_best theta_dim"]]:
        """Match a single voxel signal to the library.

        Returns
        -------
        best_params : ``(theta_dim,)``
        topk_params : ``(k_best, theta_dim)``
        """
        # Normalise query
        s_norm = signal / jnp.maximum(jnp.linalg.norm(signal), 1e-12)

        # Cosine similarity: (N,) = (N, M) @ (M,)
        cos_sim = self._lib_signals_normed @ s_norm

        if penalty is not None and self.alpha > 0:
            cos_sim = cos_sim - self.alpha * penalty

        # Top-K
        topk_vals, topk_idx = jax.lax.top_k(cos_sim, self.k_best)
        topk_params = self.library.params[topk_idx]
        best_params = topk_params[0]

        return best_params, topk_params

    def match_batch(
        self,
        signals: Float[Array, "V M"],
        penalty: Optional[Float[Array, "N"]] = None,
    ) -> Tuple[Float[Array, "V theta_dim"], Float[Array, "V k_best theta_dim"]]:
        """Batched matching over multiple voxels (vmapped)."""
        return jax.vmap(self.match_single, in_axes=(0, None))(signals, penalty)

    # ------------------------------------------------------------------ #
    # Volume-level matching
    # ------------------------------------------------------------------ #

    def match_volume(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        batch_size: int = 4096,
        extract_metrics: bool = False,
    ) -> dict:
        """Match every voxel in a 4-D volume to the library.

        Parameters
        ----------
        data : np.ndarray ``(X, Y, Z, M)``
        mask : np.ndarray ``(X, Y, Z)`` optional bool mask.
        batch_size : int
        extract_metrics : bool
            If True, also run ``MultiMetricExtractor`` on the results.

        Returns
        -------
        results : dict  ``{param_name: np.ndarray (X, Y, Z)}``
        """
        vol_shape = data.shape[:3]
        if mask is None:
            mask = np.ones(vol_shape, dtype=bool)

        flat = data[mask]
        n_vox = flat.shape[0]
        n_batches = math.ceil(n_vox / batch_size)

        best_all = []
        for i in range(n_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, n_vox)
            batch = jnp.array(flat[s:e])
            best, _ = self.match_batch(batch)
            best_all.append(np.asarray(best))

        best_concat = np.concatenate(best_all, axis=0)

        results = {}
        for j, name in enumerate(self.library.parameter_names):
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = best_concat[:, j]
            results[name] = vol

        # Uncertainty: IQR from top-K
        # (omitted in volume path for speed; available via match_batch)

        if extract_metrics:
            try:
                from dmipy_jax.pipeline.metrics import MultiMetricExtractor
                extractor = MultiMetricExtractor(self.library.parameter_names)
                metric_maps = extractor.extract(results, vol_shape)
                results.update(metric_maps)
            except ImportError:
                pass

        return results

    # ------------------------------------------------------------------ #
    # Uncertainty from top-K
    # ------------------------------------------------------------------ #

    @staticmethod
    def topk_iqr(
        topk_params: Float[Array, "... k_best theta_dim"],
    ) -> Float[Array, "... theta_dim"]:
        """Inter-quartile range across the top-K matched parameters."""
        q75 = jnp.percentile(topk_params, 75, axis=-2)
        q25 = jnp.percentile(topk_params, 25, axis=-2)
        return q75 - q25
