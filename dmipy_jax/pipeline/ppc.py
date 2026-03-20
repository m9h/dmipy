"""
Posterior Predictive Checks (PPC) for SBI model diagnostics.

Verifies that signals simulated from posterior samples are consistent
with the observed data.  Complements SBC (calibration) by checking
data-space fit quality.
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
# PPC Result
# --------------------------------------------------------------------------- #


@dataclass
class PPCResult:
    """Results from posterior predictive checks.

    Attributes
    ----------
    rmse_per_obs : np.ndarray
        Root-mean-square error between observed signal and the posterior
        predictive mean, shape ``(n_checks,)``.
    coverage_per_measurement : np.ndarray
        Fraction of observations where x* falls within the 90 % prediction
        interval, shape ``(n_measurements,)``.
    chi_squared : np.ndarray
        Chi-squared statistic per observation, shape ``(n_checks,)``.
    n_measurements : int
        Number of signal measurements (M).
    """

    rmse_per_obs: np.ndarray
    coverage_per_measurement: np.ndarray
    chi_squared: np.ndarray
    n_measurements: int

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #

    @property
    def mean_rmse(self) -> float:
        """Mean RMSE across all observations."""
        return float(np.mean(self.rmse_per_obs))

    @property
    def median_rmse(self) -> float:
        """Median RMSE across all observations."""
        return float(np.median(self.rmse_per_obs))

    @property
    def mean_coverage(self) -> float:
        """Mean 90 % coverage across all measurement indices."""
        return float(np.mean(self.coverage_per_measurement))

    @property
    def reduced_chi_squared(self) -> float:
        """Mean chi-squared / n_measurements.  Should be ~1 for good fit."""
        return float(np.mean(self.chi_squared)) / self.n_measurements

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        """Print a human-readable summary of PPC results."""
        print("Posterior Predictive Check Results")
        print("=" * 50)
        print(f"  Mean RMSE:             {self.mean_rmse:.4f}")
        print(f"  Median RMSE:           {self.median_rmse:.4f}")
        print(f"  Mean coverage (90%):   {self.mean_coverage:.1%}")
        print(f"  Reduced chi-squared:   {self.reduced_chi_squared:.2f}  (target: ~1.0)")

    def plot_prediction_intervals(
        self,
        observed_signals: Optional[np.ndarray] = None,
        predicted_signals: Optional[np.ndarray] = None,
        n_examples: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot observed vs predicted signals for a few examples.

        Parameters
        ----------
        observed_signals : np.ndarray, optional
            Shape ``(n_checks, n_measurements)``.  The observed signals.
        predicted_signals : np.ndarray, optional
            Shape ``(n_checks, n_posterior_samples, n_measurements)``.
            Predicted signals from posterior samples.
        n_examples : int
            Number of example observations to plot.
        save_path : str, optional
            If provided, save the figure to this path instead of displaying.
        """
        import matplotlib

        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if observed_signals is None or predicted_signals is None:
            print(
                "plot_prediction_intervals requires observed_signals and "
                "predicted_signals arrays.  Re-run PPCDiagnostic.run() with "
                "return_signals=True (not yet implemented) or store them "
                "externally."
            )
            return

        n_examples = min(n_examples, len(observed_signals))
        fig, axes = plt.subplots(
            1, n_examples, figsize=(4 * n_examples, 3), squeeze=False
        )

        for i in range(n_examples):
            ax = axes[0, i]
            x_obs = observed_signals[i]
            x_pred = predicted_signals[i]  # (L, M)
            m_idx = np.arange(len(x_obs))

            lo = np.percentile(x_pred, 5, axis=0)
            hi = np.percentile(x_pred, 95, axis=0)
            med = np.median(x_pred, axis=0)

            ax.fill_between(m_idx, lo, hi, alpha=0.3, label="90% PI")
            ax.plot(m_idx, med, "b-", linewidth=1, label="Median pred")
            ax.plot(m_idx, x_obs, "r.", markersize=3, label="Observed")
            ax.set_title(f"Obs {i}")
            ax.set_xlabel("Measurement")
            if i == 0:
                ax.set_ylabel("Signal")
                ax.legend(fontsize=7)

        fig.suptitle("Posterior Predictive Checks", fontsize=14)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved PPC plot to {save_path}")
            plt.close(fig)
        else:
            plt.show()


# --------------------------------------------------------------------------- #
# PPC Diagnostic Runner
# --------------------------------------------------------------------------- #


class PPCDiagnostic:
    """Run posterior predictive checks on a trained SBI model.

    Parameters
    ----------
    model : eqx.Module
        Trained ``_NormalisedMDN`` or ``_NormalisedFlow``.
    simulator : ModelSimulator
        Provides the forward model for re-simulation.
    n_posterior_samples : int
        Number of posterior samples per observation (L).
    """

    def __init__(
        self,
        model,
        simulator: ModelSimulator,
        n_posterior_samples: int = 200,
    ):
        self.model = model
        self.simulator = simulator
        self.n_posterior_samples = n_posterior_samples

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, key: jax.Array, n_checks: int = 500) -> PPCResult:
        """Execute the full PPC loop.

        For each check:
        1. Sample theta* from the prior and simulate a noisy observation x*.
        2. Draw L posterior samples from the model given x*.
        3. Re-simulate clean signals from each posterior sample.
        4. Compute RMSE, coverage, and chi-squared.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_checks : int
            Number of observations to check (N).

        Returns
        -------
        PPCResult
        """
        rmse_list = []
        chi2_list = []
        # Accumulate per-measurement coverage counts
        n_meas = self.simulator.signal_dim
        coverage_counts = np.zeros(n_meas, dtype=np.float64)

        for i in range(n_checks):
            key, k_prior, k_sim, k_noise, k_post, k_resim = jax.random.split(
                key, 6
            )

            # 1. Sample theta* from the prior and generate noisy observation
            theta_star = self.simulator.prior_sampler(k_prior, 1)  # (1, D)

            signal_clean = self.simulator.simulate(k_sim, theta_star)  # (1, M)
            signal_noisy = self.simulator.add_noise(k_noise, signal_clean)  # (1, M)

            # b0-normalise to match training preprocessing
            b0_mask = self.simulator.acquisition.bvalues < 100e6
            if jnp.any(b0_mask):
                b0_mean = jnp.mean(
                    signal_noisy[:, b0_mask], axis=1, keepdims=True
                )
                b0_mean = jnp.maximum(b0_mean, 1e-6)
                signal_noisy = signal_noisy / b0_mean

            x_star = signal_noisy[0]  # (M,)

            # 2. Draw posterior samples
            posterior_samples = self._sample_posterior(
                x_star, k_post, self.n_posterior_samples
            )  # (L, D)

            # 3. Re-simulate clean signals from posterior samples
            x_pred = self.simulator.simulate(
                k_resim, posterior_samples
            )  # (L, M)

            # b0-normalise predicted signals identically
            if jnp.any(b0_mask):
                b0_pred = jnp.mean(x_pred[:, b0_mask], axis=1, keepdims=True)
                b0_pred = jnp.maximum(b0_pred, 1e-6)
                x_pred = x_pred / b0_pred

            x_pred_np = np.asarray(x_pred)
            x_star_np = np.asarray(x_star)

            # 4a. RMSE: ||x* - E[x_pred]||
            x_pred_mean = np.mean(x_pred_np, axis=0)  # (M,)
            rmse = float(np.sqrt(np.mean((x_star_np - x_pred_mean) ** 2)))
            rmse_list.append(rmse)

            # 4b. Coverage: per-measurement, does x* lie in [5th, 95th]?
            lo = np.percentile(x_pred_np, 5, axis=0)  # (M,)
            hi = np.percentile(x_pred_np, 95, axis=0)  # (M,)
            inside = (x_star_np >= lo) & (x_star_np <= hi)
            coverage_counts += inside.astype(np.float64)

            # 4c. Chi-squared: sum((x* - x_pred_mean)^2 / var(x_pred))
            x_pred_var = np.var(x_pred_np, axis=0)  # (M,)
            # Guard against zero variance
            x_pred_var = np.maximum(x_pred_var, 1e-12)
            chi2 = float(np.sum((x_star_np - x_pred_mean) ** 2 / x_pred_var))
            chi2_list.append(chi2)

            if (i + 1) % 100 == 0:
                print(f"  PPC progress: {i + 1}/{n_checks}")

        rmse_arr = np.array(rmse_list)
        chi2_arr = np.array(chi2_list)
        coverage_per_meas = coverage_counts / n_checks

        return PPCResult(
            rmse_per_obs=rmse_arr,
            coverage_per_measurement=coverage_per_meas,
            chi_squared=chi2_arr,
            n_measurements=n_meas,
        )

    # ------------------------------------------------------------------ #
    # Posterior sampling (model-type dispatch)
    # ------------------------------------------------------------------ #

    def _sample_posterior(self, signal, key, n_samples):
        """Draw posterior samples, dispatching on model type.

        Parameters
        ----------
        signal : jax.Array, shape ``(M,)``
            Observed (noisy, b0-normalised) signal.
        key : jax.Array
            PRNG key.
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : jax.Array, shape ``(n_samples, D)``
            Posterior samples in *physical* (denormalised) parameter space.
        """
        if isinstance(self.model, _NormalisedMDN):
            return self._sample_mdn(self.model, signal, key, n_samples)
        elif isinstance(self.model, _NormalisedFlow):
            return self._sample_flow(self.model, signal, key, n_samples)
        else:
            raise TypeError(
                f"Unsupported model type: {type(self.model)}. "
                "Expected _NormalisedMDN or _NormalisedFlow."
            )

    @staticmethod
    def _sample_mdn(model: _NormalisedMDN, signal, key, n_samples):
        """Sample from an MDN posterior.

        The ``_NormalisedMDN.__call__`` already returns mu and log_sigma in
        physical units, so samples come out denormalised.
        """
        logits_pi, mu, log_sigma = model(signal)
        sigma = jnp.exp(log_sigma)
        k1, k2 = jax.random.split(key)
        indices = jax.random.categorical(k1, logits_pi, shape=(n_samples,))
        mu_sel = mu[indices]
        sigma_sel = sigma[indices]
        return mu_sel + sigma_sel * jax.random.normal(k2, mu_sel.shape)

    @staticmethod
    def _sample_flow(model: _NormalisedFlow, signal, key, n_samples):
        """Sample from a normalising-flow posterior.

        ``_NormalisedFlow.sample`` already denormalises internally.
        """
        return model.sample(key, (n_samples,), condition=signal)
