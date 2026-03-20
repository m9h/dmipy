"""
Simulation-Based Calibration (SBC) for validating SBI posteriors.

Implements the rank-histogram diagnostic from Talts et al. (2018).
If the posterior is well-calibrated, the rank of the true parameter among
posterior samples should be uniformly distributed.

References
----------
Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
Validating Bayesian inference algorithms with simulation-based calibration.
arXiv:1804.06788.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import _NormalisedMDN, _NormalisedFlow


# --------------------------------------------------------------------------- #
# SBC Result
# --------------------------------------------------------------------------- #


@dataclass
class SBCResult:
    """Holds the output of an SBC run.

    Attributes
    ----------
    ranks : np.ndarray
        Rank of the true parameter in the posterior samples, shape
        ``(n_sbc_samples, n_params)``.  Each entry is in
        ``[0, n_posterior_samples]``.
    parameter_names : list of str
        Human-readable names for each parameter dimension.
    n_posterior_samples : int
        Number of posterior draws used per SBC repetition (L).
    """

    ranks: np.ndarray
    parameter_names: List[str]
    n_posterior_samples: int

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def coverage_at(self, level: float = 0.9) -> np.ndarray:
        """Fraction of test cases where theta* is inside the credible interval.

        For each SBC sample we check whether the true parameter falls within
        the central ``level`` credible interval constructed from the posterior
        samples.  Under perfect calibration this fraction equals ``level``.

        Parameters
        ----------
        level : float
            Credible level, e.g. 0.9 for a 90 % interval.

        Returns
        -------
        coverage : np.ndarray, shape ``(n_params,)``
        """
        lower_q = (1.0 - level) / 2.0
        upper_q = (1.0 + level) / 2.0

        # Ranks encode the position of theta* in the sorted posterior.
        # theta* is inside the [lower_q, upper_q] interval iff its rank
        # (normalised) is between lower_q and upper_q.
        L = self.n_posterior_samples
        normalised = self.ranks / L  # shape (N, D)
        inside = (normalised >= lower_q) & (normalised <= upper_q)
        return np.mean(inside, axis=0)

    def ks_test(self) -> dict:
        """Kolmogorov-Smirnov test for uniformity of each parameter's ranks.

        Returns
        -------
        results : dict
            ``{param_name: {"statistic": float, "pvalue": float}}``.
        """
        from scipy.stats import kstest

        L = self.n_posterior_samples
        results = {}
        for d, name in enumerate(self.parameter_names):
            # Normalise ranks to [0, 1] for the uniform CDF test
            normalised = self.ranks[:, d] / (L + 1)
            stat, pval = kstest(normalised, "uniform")
            results[name] = {"statistic": float(stat), "pvalue": float(pval)}
        return results

    def print_summary(self) -> None:
        """Print a human-readable summary table."""
        ks = self.ks_test()
        cov90 = self.coverage_at(0.9)
        cov50 = self.coverage_at(0.5)

        hdr = f"{'Parameter':<20s} {'Cov@50%':>8s} {'Cov@90%':>8s} {'KS stat':>8s} {'KS p':>8s} {'Pass?':>6s}"
        print(hdr)
        print("-" * len(hdr))
        for d, name in enumerate(self.parameter_names):
            k = ks[name]
            passed = "yes" if k["pvalue"] > 0.05 else "NO"
            print(
                f"{name:<20s} {cov50[d]:8.3f} {cov90[d]:8.3f} "
                f"{k['statistic']:8.4f} {k['pvalue']:8.4f} {passed:>6s}"
            )

    def plot_rank_histograms(self, save_path: Optional[str] = None) -> None:
        """Plot rank histograms for each parameter dimension.

        Parameters
        ----------
        save_path : str, optional
            If provided, save the figure to this path instead of displaying.
        """
        import matplotlib

        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_params = len(self.parameter_names)
        n_cols = min(n_params, 4)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        L = self.n_posterior_samples
        n_bins = min(20, L + 1)

        for d, name in enumerate(self.parameter_names):
            r, c = divmod(d, n_cols)
            ax = axes[r, c]
            ax.hist(self.ranks[:, d], bins=n_bins, range=(0, L), density=True,
                    edgecolor="black", alpha=0.7)
            # Expected uniform level
            ax.axhline(1.0 / L, color="red", linestyle="--", linewidth=1,
                       label="Uniform")
            ax.set_title(name)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Density")

        # Hide unused axes
        for d in range(n_params, n_rows * n_cols):
            r, c = divmod(d, n_cols)
            axes[r, c].set_visible(False)

        fig.suptitle("SBC Rank Histograms", fontsize=14)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved rank histogram plot to {save_path}")
            plt.close(fig)
        else:
            plt.show()


# --------------------------------------------------------------------------- #
# SBC Diagnostic Runner
# --------------------------------------------------------------------------- #


class SBCDiagnostic:
    """Run Simulation-Based Calibration on a trained SBI model.

    Parameters
    ----------
    model : eqx.Module
        Trained model -- either a ``_NormalisedMDN`` or ``_NormalisedFlow``
        wrapper returned by ``train_sbi``.
    simulator : ModelSimulator
        The same simulator used during training (provides the prior and
        forward model).
    n_posterior_samples : int
        Number of posterior draws per SBC repetition (L).
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

    def run(self, key: jax.Array, n_sbc_samples: int = 1000) -> SBCResult:
        """Execute the full SBC loop.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        n_sbc_samples : int
            Number of SBC repetitions (N).

        Returns
        -------
        SBCResult
        """
        all_ranks = []
        for i in range(n_sbc_samples):
            key, k_prior, k_sim, k_noise, k_post = jax.random.split(key, 5)

            # 1. Sample theta* from the prior
            theta_star = self.simulator.prior_sampler(k_prior, 1)  # (1, D)
            theta_star_flat = theta_star[0]  # (D,)

            # 2. Simulate and add noise
            signal_clean = self.simulator.simulate(k_sim, theta_star)  # (1, S)
            signal_noisy = self.simulator.add_noise(k_noise, signal_clean)  # (1, S)

            # b0-normalise to match training preprocessing
            b0_mask = self.simulator.acquisition.bvalues < 100e6
            if jnp.any(b0_mask):
                b0_mean = jnp.mean(signal_noisy[:, b0_mask], axis=1, keepdims=True)
                b0_mean = jnp.maximum(b0_mean, 1e-6)
                signal_noisy = signal_noisy / b0_mean

            x_star = signal_noisy[0]  # (S,)

            # 3. Draw posterior samples
            posterior_samples = self._sample_posterior(
                x_star, k_post, self.n_posterior_samples
            )  # (L, D)

            # 4. Compute ranks
            ranks = compute_ranks(theta_star_flat, posterior_samples)
            all_ranks.append(np.asarray(ranks))

            if (i + 1) % 100 == 0:
                print(f"  SBC progress: {i + 1}/{n_sbc_samples}")

        ranks_array = np.stack(all_ranks, axis=0)  # (N, D)

        return SBCResult(
            ranks=ranks_array,
            parameter_names=list(self.simulator.parameter_names),
            n_posterior_samples=self.n_posterior_samples,
        )

    # ------------------------------------------------------------------ #
    # Posterior sampling (model-type dispatch)
    # ------------------------------------------------------------------ #

    def _sample_posterior(self, signal, key, n_samples):
        """Draw posterior samples, dispatching on model type.

        Parameters
        ----------
        signal : jax.Array, shape ``(S,)``
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


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #


def compute_ranks(theta_true, posterior_samples):
    """Compute the rank of theta_true among posterior_samples.

    Parameters
    ----------
    theta_true : jax.Array, shape ``(D,)``
        True parameter vector.
    posterior_samples : jax.Array, shape ``(L, D)``
        Posterior draws.

    Returns
    -------
    ranks : jax.Array, shape ``(D,)``
        Per-dimension rank (number of posterior samples < theta_true).
    """
    return jnp.sum(posterior_samples < theta_true[None, :], axis=0)
