"""
Conformal prediction intervals for SBI parameter estimates.

Provides distribution-free, finite-sample valid coverage guarantees
that work regardless of whether the MDN/flow posterior is well-calibrated.

Split conformal inference (Vovk et al., 2005; Angelopoulos & Bates, 2023):
1. On a held-out calibration set, compute nonconformity scores.
2. Take the (1-alpha)(1+1/n)-quantile of the scores.
3. Form prediction intervals by adding/subtracting that quantile.

Two methods are supported:
- "absolute": simple symmetric intervals around the posterior mean.
- "cqr": Conformalized Quantile Regression -- uses MDN/flow quantiles
  for adaptive (heteroscedastic) intervals.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
Random World. Springer.

Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile
Regression. NeurIPS.

Angelopoulos, A. N. & Bates, S. (2023). Conformal Prediction: A Gentle
Introduction. Foundations and Trends in Machine Learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from dmipy_jax.pipeline.train import _NormalisedMDN, _NormalisedFlow


# --------------------------------------------------------------------------- #
# Conformal Result
# --------------------------------------------------------------------------- #


@dataclass
class ConformalResult:
    """Holds the output of a conformal calibration run.

    Attributes
    ----------
    quantiles : np.ndarray
        Shape ``(theta_dim,)`` -- per-parameter calibration quantiles q-hat.
    alpha : float
        Miscoverage rate used for calibration.
    method : str
        ``"absolute"`` or ``"cqr"``.
    n_calibration : int
        Number of calibration samples used.
    parameter_names : list of str
        Human-readable names for each parameter dimension.
    empirical_coverage : float
        Coverage on the calibration set (should be >= 1 - alpha).
    """

    quantiles: np.ndarray
    alpha: float
    method: str
    n_calibration: int
    parameter_names: List[str]
    empirical_coverage: float

    def print_summary(self) -> None:
        """Print a human-readable summary of the calibration result."""
        hdr = (
            f"{'Parameter':<20s} {'q-hat':>10s} {'Target cov':>12s}"
        )
        print("Conformal Calibration Summary")
        print(f"  Method:              {self.method}")
        print(f"  Alpha:               {self.alpha}")
        print(f"  Target coverage:     {1 - self.alpha:.1%}")
        print(f"  Empirical coverage:  {self.empirical_coverage:.1%}")
        print(f"  Calibration samples: {self.n_calibration}")
        print()
        print(hdr)
        print("-" * len(hdr))
        for d, name in enumerate(self.parameter_names):
            print(
                f"{name:<20s} {self.quantiles[d]:10.4f} "
                f"{1 - self.alpha:12.1%}"
            )


# --------------------------------------------------------------------------- #
# Conformal Calibrator
# --------------------------------------------------------------------------- #


class ConformalCalibrator:
    """Calibrate prediction intervals using split conformal inference.

    Parameters
    ----------
    model : eqx.Module
        Trained ``_NormalisedMDN`` or ``_NormalisedFlow``.
    parameter_names : list of str
        Ordered parameter names matching the theta dimension.
    n_posterior_samples : int
        Number of posterior samples used for computing the MDN/flow
        posterior mean and quantiles (for CQR).  Default 500.
    """

    def __init__(
        self,
        model: eqx.Module,
        parameter_names: List[str],
        n_posterior_samples: int = 500,
    ):
        self.model = model
        self.parameter_names = list(parameter_names)
        self.n_posterior_samples = n_posterior_samples

        # Populated after calibration
        self._result: Optional[ConformalResult] = None
        self._method: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        theta_cal: np.ndarray,
        signals_cal: np.ndarray,
        alpha: float = 0.1,
        method: str = "cqr",
    ) -> ConformalResult:
        """Run the calibration phase on held-out calibration data.

        Parameters
        ----------
        theta_cal : array, shape ``(n_cal, theta_dim)``
            Ground-truth parameters for the calibration set.
        signals_cal : array, shape ``(n_cal, signal_dim)``
            Corresponding observed signals (already noised / normalised).
        alpha : float
            Miscoverage rate.  Default 0.1 gives 90% coverage.
        method : str
            ``"absolute"`` for simple symmetric intervals, or ``"cqr"``
            for Conformalized Quantile Regression.

        Returns
        -------
        ConformalResult
        """
        theta_cal = np.asarray(theta_cal)
        signals_cal = np.asarray(signals_cal)
        n_cal = theta_cal.shape[0]
        theta_dim = theta_cal.shape[1]

        if method not in ("absolute", "cqr"):
            raise ValueError(
                f"Unknown method {method!r}; expected 'absolute' or 'cqr'."
            )

        # Conformal quantile level: ceil((1-alpha)(1+1/n)) / 1 clipped to [0,1]
        quantile_level = min(
            (1 - alpha) * (1 + 1 / n_cal), 1.0
        )

        if method == "absolute":
            result = self._calibrate_absolute(
                theta_cal, signals_cal, alpha, quantile_level
            )
        else:
            result = self._calibrate_cqr(
                theta_cal, signals_cal, alpha, quantile_level
            )

        self._result = result
        self._method = method
        self._alpha = alpha
        return result

    def predict_intervals(
        self,
        signals: np.ndarray,
        alpha: Optional[float] = None,
    ) -> dict:
        """Compute conformal prediction intervals for new observations.

        Parameters
        ----------
        signals : array, shape ``(batch, signal_dim)``
            Observed signals.
        alpha : float, optional
            If provided, must match the alpha used during calibration
            (reserved for future multi-alpha support).

        Returns
        -------
        result : dict
            Keys ``"mean"``, ``"lower"``, ``"upper"``, ``"width"`` each
            with shape ``(batch, theta_dim)``.
        """
        if self._result is None:
            raise RuntimeError(
                "Must call calibrate() before predict_intervals()."
            )
        if alpha is not None and alpha != self._result.alpha:
            raise ValueError(
                f"alpha={alpha} does not match calibration alpha="
                f"{self._result.alpha}.  Re-run calibrate() with the "
                f"desired alpha."
            )

        signals = np.asarray(signals)
        q_hat = self._result.quantiles  # (theta_dim,)

        if self._method == "absolute":
            means = self._compute_posterior_means(signals)
            lower = means - q_hat[None, :]
            upper = means + q_hat[None, :]
        else:  # cqr
            means, q_lo, q_hi = self._compute_posterior_quantiles(
                signals, self._alpha
            )
            lower = q_lo - q_hat[None, :]
            upper = q_hi + q_hat[None, :]

        return {
            "mean": means,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
        }

    # ------------------------------------------------------------------ #
    # Absolute method
    # ------------------------------------------------------------------ #

    def _calibrate_absolute(self, theta_cal, signals_cal, alpha, quantile_level):
        """Calibrate using simple absolute residuals."""
        n_cal = theta_cal.shape[0]
        theta_dim = theta_cal.shape[1]

        # Compute posterior means for all calibration samples
        means = self._compute_posterior_means(signals_cal)  # (n_cal, theta_dim)

        # Per-parameter absolute residuals
        residuals = np.abs(theta_cal - means)  # (n_cal, theta_dim)

        # Compute quantile per parameter
        quantiles = np.array([
            np.quantile(residuals[:, d], quantile_level)
            for d in range(theta_dim)
        ])

        # Compute empirical marginal coverage on calibration set
        lower = means - quantiles[None, :]
        upper = means + quantiles[None, :]
        per_param = (theta_cal >= lower) & (theta_cal <= upper)
        empirical_coverage = float(np.min(np.mean(per_param, axis=0)))

        return ConformalResult(
            quantiles=quantiles,
            alpha=alpha,
            method="absolute",
            n_calibration=n_cal,
            parameter_names=self.parameter_names,
            empirical_coverage=empirical_coverage,
        )

    # ------------------------------------------------------------------ #
    # CQR method
    # ------------------------------------------------------------------ #

    def _calibrate_cqr(self, theta_cal, signals_cal, alpha, quantile_level):
        """Calibrate using Conformalized Quantile Regression."""
        n_cal = theta_cal.shape[0]
        theta_dim = theta_cal.shape[1]

        # Compute posterior quantile intervals for all calibration samples
        means, q_lo, q_hi = self._compute_posterior_quantiles(
            signals_cal, alpha
        )

        # CQR nonconformity scores: how much the true value exceeds the
        # predicted interval (negative means inside the interval)
        scores = np.maximum(
            q_lo - theta_cal,   # true value below lower bound
            theta_cal - q_hi,   # true value above upper bound
        )  # (n_cal, theta_dim)

        # Compute quantile per parameter
        quantiles = np.array([
            np.quantile(scores[:, d], quantile_level)
            for d in range(theta_dim)
        ])

        # Compute empirical marginal coverage on calibration set
        lower = q_lo - quantiles[None, :]
        upper = q_hi + quantiles[None, :]
        per_param = (theta_cal >= lower) & (theta_cal <= upper)
        empirical_coverage = float(np.min(np.mean(per_param, axis=0)))

        return ConformalResult(
            quantiles=quantiles,
            alpha=alpha,
            method="cqr",
            n_calibration=n_cal,
            parameter_names=self.parameter_names,
            empirical_coverage=empirical_coverage,
        )

    # ------------------------------------------------------------------ #
    # Model inference helpers
    # ------------------------------------------------------------------ #

    def _compute_posterior_means(self, signals: np.ndarray) -> np.ndarray:
        """Compute posterior mean for each signal.

        Parameters
        ----------
        signals : array, shape ``(n, signal_dim)``

        Returns
        -------
        means : np.ndarray, shape ``(n, theta_dim)``
        """
        model = self.model

        if isinstance(model, _NormalisedMDN):
            return self._mdn_posterior_means(model, signals)
        elif isinstance(model, _NormalisedFlow):
            return self._flow_posterior_means(model, signals)
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                "Expected _NormalisedMDN or _NormalisedFlow."
            )

    def _compute_posterior_quantiles(
        self, signals: np.ndarray, alpha: float
    ) -> tuple:
        """Compute posterior mean, lower quantile, and upper quantile.

        Parameters
        ----------
        signals : array, shape ``(n, signal_dim)``
        alpha : float
            Miscoverage level.  Quantiles at alpha/2 and 1-alpha/2.

        Returns
        -------
        means : np.ndarray, shape ``(n, theta_dim)``
        q_lo : np.ndarray, shape ``(n, theta_dim)``
        q_hi : np.ndarray, shape ``(n, theta_dim)``
        """
        model = self.model

        if isinstance(model, _NormalisedMDN):
            return self._mdn_posterior_quantiles(model, signals, alpha)
        elif isinstance(model, _NormalisedFlow):
            return self._flow_posterior_quantiles(model, signals, alpha)
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                "Expected _NormalisedMDN or _NormalisedFlow."
            )

    # -- MDN helpers --------------------------------------------------- #

    def _mdn_posterior_means(self, model, signals):
        """Weighted mean of the MDN mixture components."""
        signals_jax = jnp.asarray(signals)

        @eqx.filter_jit
        def _single_mean(x):
            logits_pi, mu, _log_sigma = model(x)
            weights = jax.nn.softmax(logits_pi)
            return jnp.sum(weights[:, None] * mu, axis=0)

        means = jax.vmap(_single_mean)(signals_jax)
        return np.asarray(means)

    def _mdn_posterior_quantiles(self, model, signals, alpha):
        """Sample from the MDN and compute empirical quantiles."""
        n = signals.shape[0]
        signals_jax = jnp.asarray(signals)
        n_samples = self.n_posterior_samples

        # Sample from MDN for all signals
        key = jax.random.PRNGKey(0)

        @eqx.filter_jit
        def _single_stats(x, k):
            logits_pi, mu, log_sigma = model(x)
            sigma = jnp.exp(log_sigma)
            k1, k2 = jax.random.split(k)
            indices = jax.random.categorical(k1, logits_pi, shape=(n_samples,))
            mu_sel = mu[indices]
            sigma_sel = sigma[indices]
            samples = mu_sel + sigma_sel * jax.random.normal(k2, mu_sel.shape)

            mean = jnp.sum(
                jax.nn.softmax(logits_pi)[:, None] * mu, axis=0
            )
            q_lo = jnp.percentile(samples, 100 * alpha / 2, axis=0)
            q_hi = jnp.percentile(samples, 100 * (1 - alpha / 2), axis=0)
            return mean, q_lo, q_hi

        keys = jax.random.split(key, n)
        means, q_lo, q_hi = jax.vmap(_single_stats)(signals_jax, keys)
        return np.asarray(means), np.asarray(q_lo), np.asarray(q_hi)

    # -- Flow helpers -------------------------------------------------- #

    def _flow_posterior_means(self, model, signals):
        """Sample from the flow and take the mean."""
        signals_jax = jnp.asarray(signals)
        n_samples = self.n_posterior_samples

        @eqx.filter_jit
        def _single_mean(x, k):
            samples = model.sample(k, (n_samples,), condition=x)
            return jnp.mean(samples, axis=0)

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, signals.shape[0])
        means = jax.vmap(_single_mean)(signals_jax, keys)
        return np.asarray(means)

    def _flow_posterior_quantiles(self, model, signals, alpha):
        """Sample from the flow and compute empirical quantiles."""
        signals_jax = jnp.asarray(signals)
        n_samples = self.n_posterior_samples

        @eqx.filter_jit
        def _single_stats(x, k):
            samples = model.sample(k, (n_samples,), condition=x)
            mean = jnp.mean(samples, axis=0)
            q_lo = jnp.percentile(samples, 100 * alpha / 2, axis=0)
            q_hi = jnp.percentile(samples, 100 * (1 - alpha / 2), axis=0)
            return mean, q_lo, q_hi

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, signals.shape[0])
        means, q_lo, q_hi = jax.vmap(_single_stats)(signals_jax, keys)
        return np.asarray(means), np.asarray(q_lo), np.asarray(q_hi)


# --------------------------------------------------------------------------- #
# Functional API (model-free, for testing or external use)
# --------------------------------------------------------------------------- #


def calibrate_from_predictions(
    theta_true: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0.1,
    parameter_names: Optional[List[str]] = None,
    *,
    pred_lower: Optional[np.ndarray] = None,
    pred_upper: Optional[np.ndarray] = None,
) -> ConformalResult:
    """Calibrate conformal intervals from pre-computed predictions.

    This is the model-free entry point: pass in ground-truth parameters
    and model predictions (and optionally predicted quantile bounds for CQR).

    Parameters
    ----------
    theta_true : array, shape ``(n, theta_dim)``
        Ground-truth calibration parameters.
    predictions : array, shape ``(n, theta_dim)``
        Point predictions (posterior means).
    alpha : float
        Miscoverage rate.
    parameter_names : list of str, optional
        Names for each parameter.
    pred_lower, pred_upper : array, optional
        If both provided, use CQR method with these as the predicted
        lower/upper quantile bounds.  Otherwise use the absolute method.

    Returns
    -------
    ConformalResult
    """
    theta_true = np.asarray(theta_true)
    predictions = np.asarray(predictions)
    n_cal = theta_true.shape[0]
    theta_dim = theta_true.shape[1]

    if parameter_names is None:
        parameter_names = [f"param_{d}" for d in range(theta_dim)]

    quantile_level = min((1 - alpha) * (1 + 1 / n_cal), 1.0)

    use_cqr = pred_lower is not None and pred_upper is not None

    if use_cqr:
        pred_lower = np.asarray(pred_lower)
        pred_upper = np.asarray(pred_upper)

        # CQR scores
        scores = np.maximum(
            pred_lower - theta_true,
            theta_true - pred_upper,
        )
        quantiles = np.array([
            np.quantile(scores[:, d], quantile_level)
            for d in range(theta_dim)
        ])

        lower = pred_lower - quantiles[None, :]
        upper = pred_upper + quantiles[None, :]
        method = "cqr"
    else:
        residuals = np.abs(theta_true - predictions)
        quantiles = np.array([
            np.quantile(residuals[:, d], quantile_level)
            for d in range(theta_dim)
        ])

        lower = predictions - quantiles[None, :]
        upper = predictions + quantiles[None, :]
        method = "absolute"

    per_param = (theta_true >= lower) & (theta_true <= upper)
    empirical_coverage = float(np.min(np.mean(per_param, axis=0)))

    return ConformalResult(
        quantiles=quantiles,
        alpha=alpha,
        method=method,
        n_calibration=n_cal,
        parameter_names=parameter_names,
        empirical_coverage=empirical_coverage,
    )


def predict_intervals_from_predictions(
    predictions: np.ndarray,
    result: ConformalResult,
    *,
    pred_lower: Optional[np.ndarray] = None,
    pred_upper: Optional[np.ndarray] = None,
) -> dict:
    """Form prediction intervals from pre-computed predictions and a
    calibration result.

    Parameters
    ----------
    predictions : array, shape ``(batch, theta_dim)``
        Point predictions (posterior means).
    result : ConformalResult
        Output of ``calibrate_from_predictions`` or
        ``ConformalCalibrator.calibrate``.
    pred_lower, pred_upper : array, optional
        Required if ``result.method == "cqr"``.

    Returns
    -------
    dict with keys ``"mean"``, ``"lower"``, ``"upper"``, ``"width"``.
    """
    predictions = np.asarray(predictions)
    q_hat = result.quantiles

    if result.method == "cqr":
        if pred_lower is None or pred_upper is None:
            raise ValueError(
                "pred_lower and pred_upper are required for CQR intervals."
            )
        pred_lower = np.asarray(pred_lower)
        pred_upper = np.asarray(pred_upper)
        lower = pred_lower - q_hat[None, :]
        upper = pred_upper + q_hat[None, :]
    else:
        lower = predictions - q_hat[None, :]
        upper = predictions + q_hat[None, :]

    return {
        "mean": predictions,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
    }
