"""
NIfTI-in / NIfTI-out deployment for trained SBI models.
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
from dmipy_jax.pipeline.checkpoint import load_checkpoint
from dmipy_jax.inference.mdn import MixtureDensityNetwork


class SBIPredictor:
    """Load a trained SBI model and run inference on NIfTI volumes.

    Parameters
    ----------
    model : eqx.Module
        Trained MDN (or flow) from ``load_checkpoint`` or ``train_sbi``.
    config : SBIPipelineConfig
        Associated pipeline config.
    """

    def __init__(self, model: eqx.Module, config: SBIPipelineConfig):
        self.model = model
        self.config = config

    @classmethod
    def from_checkpoint(cls, path: str, *, key=None) -> "SBIPredictor":
        model, config = load_checkpoint(path, key=key)
        return cls(model, config)

    def predict_volume(
        self,
        dwi_path: str,
        bval_path: str,
        bvec_path: str,
        mask_path: Optional[str] = None,
        output_dir: str = ".",
        batch_size: int = 4096,
        extract_metrics: bool = False,
    ) -> dict:
        """Run inference on a full DWI volume.

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
        extract_metrics : bool
            If True, also compute derived DTI / NODDI metrics via
            ``MultiMetricExtractor`` (Phase 4).

        Returns
        -------
        results : dict
            ``{param_name: np.ndarray}`` with 3-D maps (X, Y, Z).
        """
        import nibabel as nib
        from dmipy_jax.data.io import load_nifti, load_bvals_bvecs, save_nifti

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
        b0_mask = acq.bvalues < 100e6  # threshold for b0
        if jnp.any(b0_mask):
            b0_mean = np.mean(flat_data[:, np.asarray(b0_mask)], axis=1, keepdims=True)
            b0_mean = np.maximum(b0_mean, 1e-6)
            flat_data = flat_data / b0_mean

        # Run batched inference
        n_voxels = flat_data.shape[0]
        n_batches = math.ceil(n_voxels / batch_size)
        predictions = []

        predict_fn = self._make_predict_fn()

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_voxels)
            batch = jnp.array(flat_data[start:end])
            pred = predict_fn(batch)
            predictions.append(np.asarray(pred))

        all_preds = np.concatenate(predictions, axis=0)  # (N_voxels, theta_dim)

        # Un-flatten to volume
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, name in enumerate(self.config.parameter_names):
            vol = np.zeros(vol_shape, dtype=np.float32)
            vol[mask] = all_preds[:, i]
            results[name] = vol
            save_nifti(vol, affine, str(output_dir / f"{name}.nii.gz"))

        if extract_metrics:
            try:
                from dmipy_jax.pipeline.metrics import MultiMetricExtractor
                extractor = MultiMetricExtractor(self.config.parameter_names)
                metric_maps = extractor.extract(results, vol_shape)
                for mname, mdata in metric_maps.items():
                    save_nifti(mdata, affine, str(output_dir / f"{mname}.nii.gz"))
                results.update(metric_maps)
            except ImportError:
                pass  # metrics module not yet available

        return results

    def _make_predict_fn(self):
        """Return a jitted function mapping (batch, signal_dim) -> (batch, theta_dim)."""
        model = self.model

        if isinstance(model, MixtureDensityNetwork):
            @eqx.filter_jit
            def predict(signals):
                def single(x):
                    logits_pi, mu, log_sigma = model(x)
                    weights = jax.nn.softmax(logits_pi)
                    return jnp.sum(weights[:, None] * mu, axis=0)
                return jax.vmap(single)(signals)
            return predict
        else:
            # Flow-based: sample posterior and take mean
            @eqx.filter_jit
            def predict(signals):
                def single(x):
                    key = jax.random.PRNGKey(0)
                    samples = model.sample(key, condition=x, shape=(100,))
                    return jnp.mean(samples, axis=0)
                return jax.vmap(single)(signals)
            return predict
