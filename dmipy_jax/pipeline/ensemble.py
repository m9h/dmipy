"""
Ensemble inference for SBI -- train multiple models with different seeds
and combine predictions for improved calibration and uncertainty.

Deep ensembles (Lakshminarayanan et al., 2017) provide:
- Better calibrated uncertainty via disagreement between members
- More robust point estimates via averaging
- Reduced sensitivity to random initialization
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi, _NormalisedMDN, _NormalisedFlow


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_ensemble(
    config: SBIPipelineConfig,
    simulator: ModelSimulator,
    n_members: int = 5,
    *,
    key: Optional[jax.Array] = None,
    print_every: int = 500,
) -> list:
    """Train an ensemble of SBI models with different random seeds.

    Parameters
    ----------
    config : SBIPipelineConfig
    simulator : ModelSimulator
    n_members : int
        Number of ensemble members.
    key : jax.Array, optional
        PRNG key; defaults to ``config.seed``.
    print_every : int
        Logging interval passed to each ``train_sbi`` call.

    Returns
    -------
    members : list of (model, losses) tuples
        Each element is the output of ``train_sbi`` for one member.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    members = []
    for i in range(n_members):
        key, k_member = jax.random.split(key)
        print(f"\n{'=' * 50}")
        print(f"Training ensemble member {i + 1}/{n_members}")
        print(f"{'=' * 50}")
        model, losses = train_sbi(config, simulator, key=k_member, print_every=print_every)
        members.append((model, losses))

    return members


# --------------------------------------------------------------------------- #
# Ensemble predictor
# --------------------------------------------------------------------------- #

class EnsemblePredictor:
    """Combine predictions from multiple trained models.

    Parameters
    ----------
    models : list of eqx.Module
        Trained models (each a ``_NormalisedMDN`` or ``_NormalisedFlow``).
    config : SBIPipelineConfig
    """

    def __init__(self, models, config: SBIPipelineConfig):
        self.models = models
        self.config = config

    @classmethod
    def from_training(cls, members, config: SBIPipelineConfig) -> "EnsemblePredictor":
        """Create from ``train_ensemble`` output.

        Parameters
        ----------
        members : list of (model, losses)
            Output of :func:`train_ensemble`.
        config : SBIPipelineConfig
        """
        models = [m for m, _ in members]
        return cls(models, config)

    # ------------------------------------------------------------------ #
    # MDN helpers
    # ------------------------------------------------------------------ #

    def _mdn_member_stats(self, model, signals):
        """Compute per-member mean and aleatoric std for an MDN model.

        Parameters
        ----------
        model : _NormalisedMDN
        signals : (batch, signal_dim)

        Returns
        -------
        mean : (batch, theta_dim)
        aleatoric : (batch, theta_dim)
        """

        @eqx.filter_jit
        def _stats(sigs):
            def _single(x):
                logits_pi, mu, log_sigma = model(x)
                weights = jax.nn.softmax(logits_pi)  # (K,)
                sigma = jnp.exp(log_sigma)            # (K, D)

                # Weighted mean: sum_k w_k * mu_k
                mean = jnp.sum(weights[:, None] * mu, axis=0)  # (D,)

                # Law of total variance:
                # Var = E[sigma^2] + E[mu^2] - (E[mu])^2
                #     = sum_k w_k * (sigma_k^2 + mu_k^2) - mean^2
                var = (
                    jnp.sum(weights[:, None] * (sigma ** 2 + mu ** 2), axis=0)
                    - mean ** 2
                )
                aleatoric = jnp.sqrt(jnp.maximum(var, 0.0))
                return mean, aleatoric

            return jax.vmap(_single)(sigs)

        return _stats(signals)

    def _flow_member_stats(self, model, signals, n_samples: int = 200):
        """Compute per-member mean and aleatoric std for a flow model.

        Parameters
        ----------
        model : _NormalisedFlow
        signals : (batch, signal_dim)
        n_samples : int
            Number of posterior samples per voxel.

        Returns
        -------
        mean : (batch, theta_dim)
        aleatoric : (batch, theta_dim)
        """

        @eqx.filter_jit
        def _stats(sigs):
            def _single(x):
                # Use a deterministic key derived from the signal for reproducibility
                key = jax.random.PRNGKey(0)
                samples = model.sample(key, (n_samples,), condition=x)  # (n_samples, D)
                mean = jnp.mean(samples, axis=0)
                aleatoric = jnp.std(samples, axis=0)
                return mean, aleatoric

            return jax.vmap(_single)(sigs)

        return _stats(signals)

    def _member_stats(self, model, signals):
        """Dispatch to MDN or flow stats."""
        if isinstance(model, _NormalisedMDN):
            return self._mdn_member_stats(model, signals)
        elif isinstance(model, _NormalisedFlow):
            return self._flow_member_stats(model, signals)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    # ------------------------------------------------------------------ #
    # Prediction methods
    # ------------------------------------------------------------------ #

    def predict_mean(self, signals):
        """Ensemble mean prediction (average of member means).

        Parameters
        ----------
        signals : (batch, signal_dim)

        Returns
        -------
        mean : (batch, theta_dim)
            Ensemble average of individual model means.
        std : (batch, theta_dim)
            Inter-model standard deviation (epistemic uncertainty).
        """
        member_means = []
        for model in self.models:
            m, _ = self._member_stats(model, signals)
            member_means.append(m)

        # Stack: (n_members, batch, theta_dim)
        stacked = jnp.stack(member_means, axis=0)

        mean = jnp.mean(stacked, axis=0)       # (batch, theta_dim)
        std = jnp.std(stacked, axis=0)          # (batch, theta_dim)
        return mean, std

    def predict_with_uncertainty(self, signals):
        """Full uncertainty decomposition.

        Parameters
        ----------
        signals : (batch, signal_dim)

        Returns
        -------
        result : dict
            ``'mean'``      -- ensemble mean prediction ``(batch, theta_dim)``
            ``'aleatoric'`` -- mean of per-model uncertainty ``(batch, theta_dim)``
            ``'epistemic'`` -- std across model means ``(batch, theta_dim)``
            ``'total'``     -- ``sqrt(aleatoric^2 + epistemic^2)``
        """
        member_means = []
        member_aleatorics = []

        for model in self.models:
            m, a = self._member_stats(model, signals)
            member_means.append(m)
            member_aleatorics.append(a)

        means_stack = jnp.stack(member_means, axis=0)        # (M, B, D)
        aleat_stack = jnp.stack(member_aleatorics, axis=0)   # (M, B, D)

        ensemble_mean = jnp.mean(means_stack, axis=0)        # (B, D)
        epistemic = jnp.std(means_stack, axis=0)             # (B, D)
        aleatoric = jnp.mean(aleat_stack, axis=0)            # (B, D)
        total = jnp.sqrt(aleatoric ** 2 + epistemic ** 2)

        return {
            "mean": ensemble_mean,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
        }

    def predict_volume(
        self,
        dwi_path: str,
        bval_path: str,
        bvec_path: str,
        mask_path: Optional[str] = None,
        output_dir: str = ".",
        batch_size: int = 4096,
    ) -> dict:
        """Run ensemble inference on a NIfTI volume.

        The interface mirrors :meth:`SBIPredictor.predict_volume` but produces
        additional uncertainty maps (``*_aleatoric``, ``*_epistemic``,
        ``*_total``).

        Parameters
        ----------
        dwi_path : str
            Path to 4-D NIfTI DWI volume ``(X, Y, Z, N_meas)``.
        bval_path, bvec_path : str
            FSL-style b-value / b-vector text files.
        mask_path : str, optional
            Binary brain mask.
        output_dir : str
            Directory for output NIfTI parameter maps.
        batch_size : int
            Number of voxels per inference batch (controls GPU memory).

        Returns
        -------
        results : dict
            ``{param_name: np.ndarray}`` with 3-D parameter maps, plus
            uncertainty maps suffixed ``_aleatoric``, ``_epistemic``,
            ``_total``.
        """
        import nibabel as nib
        from dmipy_jax.data.io import load_bvals_bvecs, save_nifti

        # Load data
        img = nib.load(dwi_path)
        dwi_data = np.asarray(img.get_fdata(), dtype=np.float32)
        affine = img.affine
        vol_shape = dwi_data.shape[:3]

        # Load mask
        if mask_path is not None:
            mask = np.asarray(nib.load(mask_path).get_fdata(), dtype=bool)
        else:
            mask = np.ones(vol_shape, dtype=bool)

        # Flatten to (N_voxels, N_meas)
        flat_data = dwi_data[mask]

        # Normalise: divide by mean b0 signal per voxel
        acq = load_bvals_bvecs(bval_path, bvec_path)
        b0_mask = acq.bvalues < 100e6
        if jnp.any(b0_mask):
            b0_mean = np.mean(flat_data[:, np.asarray(b0_mask)], axis=1, keepdims=True)
            b0_mean = np.maximum(b0_mean, 1e-6)
            flat_data = flat_data / b0_mean

        # Run batched inference with full uncertainty
        n_voxels = flat_data.shape[0]
        n_batches = math.ceil(n_voxels / batch_size)
        all_results = {k: [] for k in ("mean", "aleatoric", "epistemic", "total")}

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_voxels)
            batch = jnp.array(flat_data[start:end])
            batch_out = self.predict_with_uncertainty(batch)
            for k in all_results:
                all_results[k].append(np.asarray(batch_out[k]))

        for k in all_results:
            all_results[k] = np.concatenate(all_results[k], axis=0)

        # Un-flatten to volume and save
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(self.config.parameter_names):
            # Mean prediction
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = all_results["mean"][:, i]
            results[name] = vol
            save_nifti(vol, affine, str(output_dir / f"{name}.nii.gz"))

            # Uncertainty maps
            for unc_type in ("aleatoric", "epistemic", "total"):
                vol_unc = np.zeros(vol_shape, dtype=np.float32)
                vol_unc[mask] = all_results[unc_type][:, i]
                map_name = f"{name}_{unc_type}"
                results[map_name] = vol_unc
                save_nifti(vol_unc, affine, str(output_dir / f"{map_name}.nii.gz"))

        return results


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def save_ensemble(members, config: SBIPipelineConfig, path: str) -> None:
    """Save ensemble to disk.

    Creates ``<path>_member_0.eqx``, ``<path>_member_1.eqx``, etc., each
    with its own JSON config sidecar.

    Parameters
    ----------
    members : list of (model, losses)
        Output of :func:`train_ensemble`.
    config : SBIPipelineConfig
    path : str
        Base path (without extension).
    """
    from dmipy_jax.pipeline.checkpoint import save_checkpoint

    for i, (model, _) in enumerate(members):
        save_checkpoint(model, config, f"{path}_member_{i}")


def load_ensemble(path: str, n_members: int, *, key=None) -> EnsemblePredictor:
    """Load ensemble from disk.

    Parameters
    ----------
    path : str
        Base path used during :func:`save_ensemble`.
    n_members : int
        Number of ensemble members to load.
    key : jax.Array, optional
        PRNG key for skeleton initialisation.

    Returns
    -------
    EnsemblePredictor
    """
    from dmipy_jax.pipeline.checkpoint import load_checkpoint

    models = []
    config = None
    for i in range(n_members):
        model, cfg = load_checkpoint(f"{path}_member_{i}", key=key)
        models.append(model)
        if config is None:
            config = cfg
    return EnsemblePredictor(models, config)
