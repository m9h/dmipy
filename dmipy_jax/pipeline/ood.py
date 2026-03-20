"""
Out-of-distribution detection for SBI models.

Flags voxels where the input signal is far from the training distribution,
indicating the model's predictions may be unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import _NormalisedMDN, _NormalisedFlow


# --------------------------------------------------------------------------- #
# OOD Result
# --------------------------------------------------------------------------- #


@dataclass
class OODResult:
    """Results from out-of-distribution detection.

    Attributes
    ----------
    reconstruction_error : np.ndarray
        RMSE between observed signal and signal re-simulated from the
        posterior mean, shape ``(n_voxels,)``.
    predictive_entropy : np.ndarray
        Entropy of the MDN mixing weights (higher = more uncertain),
        shape ``(n_voxels,)``.
    mahalanobis_distance : np.ndarray
        Mahalanobis distance of each signal from the reference
        distribution mean, shape ``(n_voxels,)``.
    is_ood : np.ndarray
        Boolean array where True indicates an OOD voxel,
        shape ``(n_voxels,)``.
    threshold_reconstruction : float
        Reconstruction error threshold used for OOD flagging.
    threshold_entropy : float
        Predictive entropy threshold used for OOD flagging.
    threshold_mahalanobis : float
        Mahalanobis distance threshold used for OOD flagging.
    """

    reconstruction_error: np.ndarray
    predictive_entropy: np.ndarray
    mahalanobis_distance: np.ndarray
    is_ood: np.ndarray
    threshold_reconstruction: float
    threshold_entropy: float
    threshold_mahalanobis: float

    @property
    def ood_fraction(self) -> float:
        """Fraction of voxels flagged as OOD."""
        return float(np.mean(self.is_ood))

    def print_summary(self) -> None:
        """Print a human-readable summary of OOD detection results."""
        n_total = len(self.is_ood)
        n_ood = int(np.sum(self.is_ood))
        print("Out-of-Distribution Detection Results")
        print("=" * 50)
        print(f"  Total voxels:          {n_total}")
        print(f"  OOD voxels:            {n_ood} ({self.ood_fraction:.1%})")
        print(f"  Reconstruction RMSE:   "
              f"mean={np.mean(self.reconstruction_error):.4f}  "
              f"median={np.median(self.reconstruction_error):.4f}")
        print(f"  Predictive entropy:    "
              f"mean={np.mean(self.predictive_entropy):.4f}  "
              f"median={np.median(self.predictive_entropy):.4f}")
        print(f"  Mahalanobis distance:  "
              f"mean={np.mean(self.mahalanobis_distance):.4f}  "
              f"median={np.median(self.mahalanobis_distance):.4f}")
        print("-" * 50)
        print(f"  Thresholds:  recon={self.threshold_reconstruction:.4f}  "
              f"entropy={self.threshold_entropy:.4f}  "
              f"mahal={self.threshold_mahalanobis:.4f}")


# --------------------------------------------------------------------------- #
# OOD Detector
# --------------------------------------------------------------------------- #


class OODDetector:
    """Detect out-of-distribution inputs for a trained SBI model.

    Parameters
    ----------
    model : eqx.Module
        Trained ``_NormalisedMDN`` or ``_NormalisedFlow``.
    simulator : ModelSimulator
        Provides the forward model for reconstruction-based detection.
    """

    def __init__(self, model, simulator: ModelSimulator):
        self.model = model
        self.simulator = simulator

        # Populated by fit()
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_cov_inv: Optional[np.ndarray] = None
        self._ref_reconstruction_errors: Optional[np.ndarray] = None
        self._ref_entropies: Optional[np.ndarray] = None
        self._ref_mahalanobis: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Fit: build reference distribution statistics
    # ------------------------------------------------------------------ #

    def fit(self, key: jax.Array, n_reference: int = 5000) -> "OODDetector":
        """Generate reference distribution statistics from in-distribution
        samples.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_reference : int
            Number of in-distribution samples to generate.

        Returns
        -------
        self
            For method chaining.
        """
        k1, k2 = jax.random.split(key)

        # 1. Generate in-distribution (theta, signals) pairs
        theta_ref, signals_ref = self.simulator.sample_and_simulate(
            k1, n_reference
        )
        signals_ref_np = np.asarray(signals_ref)

        # 2. Compute reference signal statistics for Mahalanobis distance
        self._ref_mean = np.mean(signals_ref_np, axis=0)  # (M,)
        cov = np.cov(signals_ref_np, rowvar=False)  # (M, M)
        M = cov.shape[0]
        # Regularise for numerical stability
        self._ref_cov_inv = np.linalg.inv(
            cov + 1e-6 * np.eye(M)
        )

        # 3. Compute reference OOD scores
        ref_scores = self._compute_scores(signals_ref)
        self._ref_reconstruction_errors = ref_scores["reconstruction_error"]
        self._ref_entropies = ref_scores["predictive_entropy"]
        self._ref_mahalanobis = ref_scores["mahalanobis_distance"]

        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Score: compute OOD scores for new signals
    # ------------------------------------------------------------------ #

    def score(self, signals: jax.Array) -> dict:
        """Compute OOD scores for a batch of signals.

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, signal_dim)``
            Observed (b0-normalised) signals.

        Returns
        -------
        dict
            Keys: ``reconstruction_error``, ``predictive_entropy``,
            ``mahalanobis_distance``, ``is_ood``.
        """
        if not self._fitted:
            raise RuntimeError(
                "OODDetector has not been fitted. Call fit() first."
            )
        return self._compute_scores(signals)

    # ------------------------------------------------------------------ #
    # Flag volume
    # ------------------------------------------------------------------ #

    def flag_volume(
        self,
        signals: jax.Array,
        mask: Optional[np.ndarray] = None,
        threshold_percentile: float = 95.0,
    ) -> OODResult:
        """Flag OOD voxels in a volume.

        Parameters
        ----------
        signals : jax.Array, shape ``(n_voxels, signal_dim)``
            Observed (b0-normalised) signals.
        mask : np.ndarray, optional
            Boolean mask; if provided, only score True voxels and mark
            the rest as not-OOD.
        threshold_percentile : float
            Percentile of reference distribution scores to use as the
            OOD threshold (default 95).

        Returns
        -------
        OODResult
        """
        if not self._fitted:
            raise RuntimeError(
                "OODDetector has not been fitted. Call fit() first."
            )

        # Compute thresholds from reference distribution
        thresh_recon = float(
            np.percentile(self._ref_reconstruction_errors, threshold_percentile)
        )
        thresh_entropy = float(
            np.percentile(self._ref_entropies, threshold_percentile)
        )
        thresh_mahal = float(
            np.percentile(self._ref_mahalanobis, threshold_percentile)
        )

        n_voxels = signals.shape[0]

        if mask is not None:
            mask = np.asarray(mask).ravel()[:n_voxels]
            active_idx = np.where(mask)[0]
            if len(active_idx) == 0:
                return OODResult(
                    reconstruction_error=np.zeros(n_voxels),
                    predictive_entropy=np.zeros(n_voxels),
                    mahalanobis_distance=np.zeros(n_voxels),
                    is_ood=np.zeros(n_voxels, dtype=bool),
                    threshold_reconstruction=thresh_recon,
                    threshold_entropy=thresh_entropy,
                    threshold_mahalanobis=thresh_mahal,
                )
            signals_active = signals[active_idx]
            scores = self._compute_scores(signals_active)

            recon = np.zeros(n_voxels)
            entropy = np.zeros(n_voxels)
            mahal = np.zeros(n_voxels)
            is_ood = np.zeros(n_voxels, dtype=bool)

            recon[active_idx] = scores["reconstruction_error"]
            entropy[active_idx] = scores["predictive_entropy"]
            mahal[active_idx] = scores["mahalanobis_distance"]

            # Apply thresholds to active voxels
            active_recon = scores["reconstruction_error"]
            active_entropy = scores["predictive_entropy"]
            active_mahal = scores["mahalanobis_distance"]
            is_ood[active_idx] = (
                (active_recon > thresh_recon)
                | (active_entropy > thresh_entropy)
                | (active_mahal > thresh_mahal)
            )
        else:
            scores = self._compute_scores(signals)
            recon = scores["reconstruction_error"]
            entropy = scores["predictive_entropy"]
            mahal = scores["mahalanobis_distance"]

            is_ood = (
                (recon > thresh_recon)
                | (entropy > thresh_entropy)
                | (mahal > thresh_mahal)
            )

        return OODResult(
            reconstruction_error=recon,
            predictive_entropy=entropy,
            mahalanobis_distance=mahal,
            is_ood=is_ood,
            threshold_reconstruction=thresh_recon,
            threshold_entropy=thresh_entropy,
            threshold_mahalanobis=thresh_mahal,
        )

    # ------------------------------------------------------------------ #
    # Internal scoring
    # ------------------------------------------------------------------ #

    def _compute_scores(self, signals: jax.Array) -> dict:
        """Compute all OOD scores for a batch of signals.

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, signal_dim)``

        Returns
        -------
        dict with numpy arrays
        """
        signals_jax = jnp.asarray(signals)

        # --- Reconstruction error ---
        recon_error = self._reconstruction_error(signals_jax)

        # --- Predictive entropy ---
        entropy = self._predictive_entropy(signals_jax)

        # --- Mahalanobis distance ---
        mahal = self._mahalanobis_distance(signals_jax)

        recon_np = np.asarray(recon_error)
        entropy_np = np.asarray(entropy)
        mahal_np = np.asarray(mahal)

        # Default is_ood is False (caller may override with thresholds)
        is_ood = np.zeros(recon_np.shape[0], dtype=bool)

        return {
            "reconstruction_error": recon_np,
            "predictive_entropy": entropy_np,
            "mahalanobis_distance": mahal_np,
            "is_ood": is_ood,
        }

    def _reconstruction_error(self, signals: jax.Array) -> jax.Array:
        """RMSE between observed signals and signals re-simulated from
        the posterior mean.

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, M)``

        Returns
        -------
        jax.Array, shape ``(batch,)``
        """
        # Get posterior mean for each signal
        theta_hat = self._posterior_mean(signals)  # (batch, D)

        # Re-simulate clean signal from posterior mean
        signals_hat = self.simulator.simulate(
            jax.random.PRNGKey(0), theta_hat
        )  # (batch, M)

        # b0-normalise the re-simulated signal to match input normalisation
        b0_mask = self.simulator.acquisition.bvalues < 100e6
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(signals_hat[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            signals_hat = signals_hat / b0_mean

        # RMSE per voxel
        residuals = signals - signals_hat
        rmse = jnp.sqrt(jnp.mean(residuals ** 2, axis=-1))
        return rmse

    def _predictive_entropy(self, signals: jax.Array) -> jax.Array:
        """Entropy of the MDN mixing weights (or uniform entropy for flows).

        For MDN: H = -sum(pi_k * log(pi_k)) where pi = softmax(logits).
        High entropy means the model is uncertain about which component
        best explains the data.

        For flows: returns zeros (entropy is not directly available).

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, M)``

        Returns
        -------
        jax.Array, shape ``(batch,)``
        """
        if isinstance(self.model, _NormalisedMDN):
            return self._mdn_entropy(signals)
        else:
            # For flows, entropy is not directly available
            return jnp.zeros(signals.shape[0])

    def _mdn_entropy(self, signals: jax.Array) -> jax.Array:
        """Compute mixing-weight entropy for each signal via the MDN."""

        def _single_entropy(x):
            logits_pi, _mu, _log_sigma = self.model(x)
            pi = jax.nn.softmax(logits_pi)
            # Safe log: clamp pi to avoid log(0)
            log_pi = jnp.log(jnp.maximum(pi, 1e-12))
            return -jnp.sum(pi * log_pi)

        return jax.vmap(_single_entropy)(signals)

    def _mahalanobis_distance(self, signals: jax.Array) -> jax.Array:
        """Mahalanobis distance from each signal to the reference mean.

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, M)``

        Returns
        -------
        jax.Array, shape ``(batch,)``
        """
        ref_mean = jnp.asarray(self._ref_mean)
        cov_inv = jnp.asarray(self._ref_cov_inv)

        def _single_mahal(x):
            diff = x - ref_mean  # (M,)
            return jnp.sqrt(jnp.dot(diff, jnp.dot(cov_inv, diff)))

        return jax.vmap(_single_mahal)(signals)

    def _posterior_mean(self, signals: jax.Array) -> jax.Array:
        """Compute the posterior mean for each signal.

        For MDN: weighted mean of component means.
        For flow: sample-based mean (draw samples and average).

        Parameters
        ----------
        signals : jax.Array, shape ``(batch, M)``

        Returns
        -------
        jax.Array, shape ``(batch, D)``
        """
        if isinstance(self.model, _NormalisedMDN):
            return self._mdn_posterior_mean(signals)
        elif isinstance(self.model, _NormalisedFlow):
            return self._flow_posterior_mean(signals)
        else:
            raise TypeError(
                f"Unsupported model type: {type(self.model)}. "
                "Expected _NormalisedMDN or _NormalisedFlow."
            )

    def _mdn_posterior_mean(self, signals: jax.Array) -> jax.Array:
        """Weighted mean of MDN component means."""

        def _single_mean(x):
            logits_pi, mu, _log_sigma = self.model(x)
            pi = jax.nn.softmax(logits_pi)  # (K,)
            # Weighted mean: sum_k pi_k * mu_k
            return jnp.sum(pi[:, None] * mu, axis=0)  # (D,)

        return jax.vmap(_single_mean)(signals)

    def _flow_posterior_mean(
        self, signals: jax.Array, n_samples: int = 200
    ) -> jax.Array:
        """Sample-based posterior mean for flow models."""
        batch_size = signals.shape[0]
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        def _single_mean(x, key):
            samples = self.model.sample(
                key, (n_samples,), condition=x
            )  # (n_samples, D)
            return jnp.mean(samples, axis=0)  # (D,)

        return jax.vmap(_single_mean)(signals, keys)
