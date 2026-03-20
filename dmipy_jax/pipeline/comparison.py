"""
General comparison runner: evaluate dmipy-JAX SBI against conventional
fitting methods (DIPY DTI/DKI, DIPY multi_tensor simulation) on the
same data.

Supports any combination of:
  - Synthetic phantom (generate_synthetic_hcp, DIPY multi_tensor, etc.)
  - NIfTI volume (real data)
  - Pre-trained SBI checkpoint or on-the-fly training

Produces:
  - Side-by-side parameter maps
  - Scatter plots (predicted vs ground truth or vs conventional)
  - Correlation / RMSE / SSIM summary tables
  - NIfTI outputs for visualization in FSLeyes / MRtrix
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp

from dmipy_jax.library.derived_metrics import tensor_to_fa_md, eigenvalues_to_ad_rd


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class ComparisonResult:
    """Holds parameter maps from multiple methods for comparison.

    Attributes
    ----------
    methods : dict
        ``{method_name: {metric_name: np.ndarray}}``
    ground_truth : dict or None
        ``{metric_name: np.ndarray}`` if synthetic data with known GT.
    vol_shape : tuple
        Spatial volume shape.
    mask : np.ndarray or None
    """
    methods: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    ground_truth: Optional[Dict[str, np.ndarray]] = None
    vol_shape: Optional[Tuple[int, ...]] = None
    mask: Optional[np.ndarray] = None

    def add_method(self, name: str, maps: Dict[str, np.ndarray]):
        self.methods[name] = maps

    def metrics_available(self) -> List[str]:
        """Return metric names common to all methods."""
        if not self.methods:
            return []
        sets = [set(m.keys()) for m in self.methods.values()]
        return sorted(set.intersection(*sets))

    def correlation(self, method_a: str, method_b: str, metric: str) -> float:
        a = self.methods[method_a][metric].ravel()
        b = self.methods[method_b][metric].ravel()
        if self.mask is not None:
            m = self.mask.ravel().astype(bool)
            a, b = a[m], b[m]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[valid], b[valid])[0, 1])

    def rmse(self, method: str, metric: str) -> float:
        """RMSE against ground truth."""
        if self.ground_truth is None or metric not in self.ground_truth:
            return float("nan")
        pred = self.methods[method][metric].ravel()
        gt = self.ground_truth[metric].ravel()
        if self.mask is not None:
            m = self.mask.ravel().astype(bool)
            pred, gt = pred[m], gt[m]
        valid = np.isfinite(pred) & np.isfinite(gt)
        return float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2)))

    def ssim_1d(self, method: str, metric: str) -> float:
        """SSIM against ground truth (flattened)."""
        if self.ground_truth is None or metric not in self.ground_truth:
            return float("nan")
        x = self.ground_truth[metric].ravel()
        y = self.methods[method][metric].ravel()
        if self.mask is not None:
            m = self.mask.ravel().astype(bool)
            x, y = x[m], y[m]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        c1 = (0.01 * (x.max() - x.min())) ** 2
        c2 = (0.03 * (x.max() - x.min())) ** 2
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()
        sxy = np.mean((x - mx) * (y - my))
        return float((2*mx*my + c1)*(2*sxy + c2) / ((mx**2 + my**2 + c1)*(sx**2 + sy**2 + c2)))

    def print_summary(self):
        """Print a comparison table."""
        metrics = self.metrics_available()
        method_names = list(self.methods.keys())

        if self.ground_truth:
            print("\n" + "=" * 70)
            print("Comparison vs Ground Truth")
            print("=" * 70)
            header = f"{'Metric':<10s}"
            for m in method_names:
                header += f"  {m+' RMSE':>14s}  {m+' SSIM':>10s}"
            print(header)
            print("-" * len(header))
            for metric in metrics:
                if metric not in self.ground_truth:
                    continue
                row = f"{metric:<10s}"
                for m in method_names:
                    r = self.rmse(m, metric)
                    s = self.ssim_1d(m, metric)
                    row += f"  {r:>14.4e}  {s:>10.4f}"
                print(row)

        if len(method_names) >= 2:
            print("\n" + "=" * 70)
            print("Inter-method Correlations")
            print("=" * 70)
            for i, ma in enumerate(method_names):
                for mb in method_names[i+1:]:
                    corrs = []
                    for metric in metrics:
                        r = self.correlation(ma, mb, metric)
                        corrs.append(f"{metric}={r:.4f}")
                    print(f"  {ma} vs {mb}: {', '.join(corrs)}")

    def save_niftis(self, output_dir: str, affine=None):
        """Save all method maps as NIfTIs."""
        import nibabel as nib
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if affine is None:
            affine = np.eye(4)
        for method_name, maps in self.methods.items():
            method_dir = out / method_name.replace(" ", "_")
            method_dir.mkdir(exist_ok=True)
            for metric_name, data in maps.items():
                img = nib.Nifti1Image(data.astype(np.float32), affine)
                nib.save(img, str(method_dir / f"{metric_name}.nii.gz"))
        if self.ground_truth:
            gt_dir = out / "ground_truth"
            gt_dir.mkdir(exist_ok=True)
            for metric_name, data in self.ground_truth.items():
                img = nib.Nifti1Image(data.astype(np.float32), affine)
                nib.save(img, str(gt_dir / f"{metric_name}.nii.gz"))

    def plot_scatter(self, method_a: str, method_b: str, metrics=None,
                     save_path=None):
        """Scatter plots of method_a vs method_b for each metric."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if metrics is None:
            metrics = self.metrics_available()

        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            a = self.methods[method_a][metric].ravel()
            b = self.methods[method_b][metric].ravel()
            if self.mask is not None:
                m = self.mask.ravel().astype(bool)
                a, b = a[m], b[m]
            valid = np.isfinite(a) & np.isfinite(b)
            ax.scatter(a[valid], b[valid], alpha=0.2, s=3)
            lo = min(a[valid].min(), b[valid].min())
            hi = max(a[valid].max(), b[valid].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            r = np.corrcoef(a[valid], b[valid])[0, 1]
            ax.set_xlabel(method_a)
            ax.set_ylabel(method_b)
            ax.set_title(f"{metric} (r={r:.3f})")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


# --------------------------------------------------------------------------- #
# DIPY conventional fitting
# --------------------------------------------------------------------------- #

def run_dipy_dti(data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Run DIPY's DTI fitting and return FA, MD, AD, RD maps.

    Parameters
    ----------
    data : ``(X, Y, Z, N_meas)`` or ``(N_vox, N_meas)``
    bvals : ``(N_meas,)`` in s/mm^2
    bvecs : ``(N_meas, 3)``
    mask : optional brain mask

    Returns
    -------
    dict with FA, MD, AD, RD
    """
    from dipy.reconst.dti import TensorModel
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(bvals, bvecs=bvecs)
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(data, mask=mask)

    return {
        "FA": np.asarray(dti_fit.fa),
        "MD": np.asarray(dti_fit.md),
        "AD": np.asarray(dti_fit.ad),
        "RD": np.asarray(dti_fit.rd),
    }


def run_dipy_dki(data: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Run DIPY's DKI fitting and return DTI + kurtosis maps.

    Requires multi-shell data (at least 2 non-zero b-values).
    """
    from dipy.reconst.dki import DiffusionKurtosisModel
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(bvals, bvecs=bvecs)
    dki_model = DiffusionKurtosisModel(gtab)
    dki_fit = dki_model.fit(data, mask=mask)

    return {
        "FA": np.asarray(dki_fit.fa),
        "MD": np.asarray(dki_fit.md),
        "AD": np.asarray(dki_fit.ad),
        "RD": np.asarray(dki_fit.rd),
        "MK": np.asarray(dki_fit.mk()),
        "AK": np.asarray(dki_fit.ak()),
        "RK": np.asarray(dki_fit.rk()),
    }


# --------------------------------------------------------------------------- #
# dmipy-JAX SBI fitting
# --------------------------------------------------------------------------- #

def run_dmipy_sbi(data: np.ndarray, bvals_si: np.ndarray,
                  bvecs: np.ndarray, checkpoint_path: str,
                  mask: Optional[np.ndarray] = None,
                  batch_size: int = 4096) -> Dict[str, np.ndarray]:
    """Run dmipy-JAX SBI inference from a checkpoint.

    Parameters
    ----------
    data : ``(X, Y, Z, N_meas)``
    bvals_si : ``(N_meas,)`` in s/m^2 (SI units)
    bvecs : ``(N_meas, 3)``
    checkpoint_path : path to saved checkpoint (without extension)
    mask : optional brain mask
    batch_size : voxels per inference batch

    Returns
    -------
    dict with parameter maps (names from checkpoint config)
    """
    import math
    from dmipy_jax.pipeline.checkpoint import load_checkpoint
    from dmipy_jax.pipeline.train import _NormalisedMDN, _NormalisedFlow
    import equinox as eqx

    model, config = load_checkpoint(checkpoint_path)

    vol_shape = data.shape[:3]
    if mask is None:
        mask_bool = np.ones(vol_shape, dtype=bool)
    else:
        mask_bool = mask.astype(bool)

    flat_data = data[mask_bool].astype(np.float32)

    # b0-normalise
    from dmipy_jax.acquisition import JaxAcquisition
    acq = JaxAcquisition(bvalues=jnp.array(bvals_si), gradient_directions=jnp.array(bvecs))
    b0_mask_arr = acq.bvalues < 100e6
    if jnp.any(b0_mask_arr):
        b0_mean = np.mean(flat_data[:, np.asarray(b0_mask_arr)], axis=1, keepdims=True)
        b0_mean = np.maximum(b0_mean, 1e-6)
        flat_data = flat_data / b0_mean

    # Predict
    if isinstance(model, (_NormalisedMDN,)):
        @eqx.filter_jit
        def predict_batch(signals):
            def single(x):
                logits_pi, mu, log_sigma = model(x)
                weights = jax.nn.softmax(logits_pi)
                return jnp.sum(weights[:, None] * mu, axis=0)
            return jax.vmap(single)(signals)
    else:
        @eqx.filter_jit
        def predict_batch(signals):
            def single(x):
                key = jax.random.key(0)
                samples = model.sample(key, (100,), condition=x)
                return jnp.mean(samples, axis=0)
            return jax.vmap(single)(signals)

    n_vox = flat_data.shape[0]
    n_batches = math.ceil(n_vox / batch_size)
    all_preds = []
    for i in range(n_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, n_vox)
        pred = predict_batch(jnp.array(flat_data[s:e]))
        all_preds.append(np.asarray(pred))
    preds = np.concatenate(all_preds, axis=0)

    results = {}
    for j, name in enumerate(config.parameter_names):
        vol = np.full(vol_shape, np.nan, dtype=np.float32)
        vol[mask_bool] = preds[:, j]
        results[name] = vol

    return results


# --------------------------------------------------------------------------- #
# dmipy-JAX dictionary matching
# --------------------------------------------------------------------------- #

def run_dmipy_dictionary(data: np.ndarray, bvals_si: np.ndarray,
                         bvecs: np.ndarray, library_path: str,
                         mask: Optional[np.ndarray] = None,
                         batch_size: int = 4096) -> Dict[str, np.ndarray]:
    """Run dmipy-JAX dictionary matching from a saved library.

    Parameters
    ----------
    data : ``(X, Y, Z, N_meas)``
    bvals_si : ``(N_meas,)`` SI units
    bvecs : ``(N_meas, 3)``
    library_path : path to HDF5 library
    mask : optional brain mask

    Returns
    -------
    dict with parameter maps
    """
    from dmipy_jax.library.storage import SimulationLibrary
    from dmipy_jax.library.matcher import DictionaryMatcher

    lib = SimulationLibrary.load_hdf5(library_path)
    matcher = DictionaryMatcher(lib, k_best=10)

    vol_shape = data.shape[:3]
    return matcher.match_volume(data, mask=mask, batch_size=batch_size)


# --------------------------------------------------------------------------- #
# High-level runner
# --------------------------------------------------------------------------- #

class ComparisonRunner:
    """Orchestrate multi-method comparison on the same dataset.

    Usage::

        runner = ComparisonRunner()
        runner.add_dipy_dti()
        runner.add_dipy_dki()
        runner.add_sbi(checkpoint_path="checkpoints/dti_mdn")
        result = runner.run(data, bvals, bvecs, mask=mask, ground_truth=gt)
        result.print_summary()
        result.save_niftis("comparison_output/")
    """

    def __init__(self):
        self._methods = []

    def add_dipy_dti(self, name: str = "DIPY DTI"):
        self._methods.append(("dipy_dti", name, {}))
        return self

    def add_dipy_dki(self, name: str = "DIPY DKI"):
        self._methods.append(("dipy_dki", name, {}))
        return self

    def add_sbi(self, checkpoint_path: str, name: str = "dmipy SBI",
                batch_size: int = 4096):
        self._methods.append(("sbi", name, {
            "checkpoint_path": checkpoint_path,
            "batch_size": batch_size,
        }))
        return self

    def add_dictionary(self, library_path: str, name: str = "dmipy Dict",
                       batch_size: int = 4096):
        self._methods.append(("dict", name, {
            "library_path": library_path,
            "batch_size": batch_size,
        }))
        return self

    def add_custom(self, func, name: str, **kwargs):
        """Add a custom fitting function.

        ``func(data, bvals, bvecs, mask=mask, **kwargs) -> dict``
        """
        self._methods.append(("custom", name, {"func": func, **kwargs}))
        return self

    def run(
        self,
        data: np.ndarray,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        ground_truth: Optional[Dict[str, np.ndarray]] = None,
        bvals_unit: str = "si",
    ) -> ComparisonResult:
        """Run all registered methods on the same data.

        Parameters
        ----------
        data : ``(X, Y, Z, N_meas)``
        bvals : ``(N_meas,)``
        bvecs : ``(N_meas, 3)``
        mask : optional brain mask
        ground_truth : optional ``{metric: array}``
        bvals_unit : "si" (s/m^2) or "fsl" (s/mm^2)

        Returns
        -------
        ComparisonResult
        """
        bvals = np.asarray(bvals)
        bvecs = np.asarray(bvecs)

        # Unit conversion
        if bvals_unit == "si":
            bvals_si = bvals
            bvals_fsl = bvals / 1e6
        else:
            bvals_fsl = bvals
            bvals_si = bvals * 1e6

        result = ComparisonResult(
            ground_truth=ground_truth,
            vol_shape=data.shape[:3],
            mask=mask,
        )

        for method_type, name, kwargs in self._methods:
            print(f"\n--- Running: {name} ---")
            t0 = time.time()

            if method_type == "dipy_dti":
                maps = run_dipy_dti(data, bvals_fsl, bvecs, mask=mask)

            elif method_type == "dipy_dki":
                maps = run_dipy_dki(data, bvals_fsl, bvecs, mask=mask)

            elif method_type == "sbi":
                maps = run_dmipy_sbi(
                    data, bvals_si, bvecs,
                    checkpoint_path=kwargs["checkpoint_path"],
                    mask=mask,
                    batch_size=kwargs.get("batch_size", 4096),
                )

            elif method_type == "dict":
                maps = run_dmipy_dictionary(
                    data, bvals_si, bvecs,
                    library_path=kwargs["library_path"],
                    mask=mask,
                    batch_size=kwargs.get("batch_size", 4096),
                )

            elif method_type == "custom":
                func = kwargs.pop("func")
                maps = func(data, bvals, bvecs, mask=mask, **kwargs)

            else:
                raise ValueError(f"Unknown method type: {method_type}")

            elapsed = time.time() - t0
            print(f"  {name}: {elapsed:.1f}s, {len(maps)} maps")
            result.add_method(name, maps)

        return result


# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #

def generate_dipy_multitensor_phantom(
    vol_shape: Tuple[int, int, int] = (20, 20, 5),
    bvals_fsl: Optional[np.ndarray] = None,
    bvecs: Optional[np.ndarray] = None,
    snr: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Generate a 3-zone phantom using DIPY's multi_tensor simulator.

    Zones:
      - WM (left): single fiber, FA ~ 0.7
      - Crossing (center): two fibers at 60 degrees, FA ~ 0.4
      - CSF (right): isotropic, FA ~ 0

    Returns
    -------
    data, bvals_fsl, bvecs, mask, ground_truth
    """
    from dipy.sims.voxel import multi_tensor
    from dipy.core.gradients import gradient_table

    if bvals_fsl is None:
        # Standard single-shell
        n_dirs = 30
        rng = np.random.RandomState(42)
        bvecs_gen = rng.randn(n_dirs, 3)
        bvecs_gen = bvecs_gen / np.linalg.norm(bvecs_gen, axis=1, keepdims=True)
        bvals_fsl = np.concatenate([np.zeros(2), np.full(n_dirs, 1000)])
        bvecs = np.concatenate([np.array([[1, 0, 0], [0, 1, 0]]), bvecs_gen])

    gtab = gradient_table(bvals_fsl, bvecs=bvecs)
    n_meas = len(bvals_fsl)

    nx, ny, nz = vol_shape
    data = np.zeros((*vol_shape, n_meas), dtype=np.float32)
    gt_fa = np.zeros(vol_shape, dtype=np.float32)
    gt_md = np.zeros(vol_shape, dtype=np.float32)
    gt_tissue = np.zeros(vol_shape, dtype=np.float32)

    # Eigenvalue sets
    wm_evals = np.array([1.7e-3, 0.3e-3, 0.3e-3])  # DIPY uses mm^2/s
    csf_evals = np.array([3.0e-3, 3.0e-3, 3.0e-3])
    crossing_evals1 = np.array([1.7e-3, 0.4e-3, 0.4e-3])
    crossing_evals2 = np.array([1.7e-3, 0.4e-3, 0.4e-3])

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if x < nx // 3:
                    # WM: single fiber along x
                    signal, _ = multi_tensor(
                        gtab, [wm_evals], S0=1.0,
                        angles=[(90, 0)],
                        fractions=[100], snr=snr,
                    )
                    gt_tissue[x, y, z] = 1
                    eigs = wm_evals
                elif x < 2 * nx // 3:
                    # Crossing: two fibers
                    signal, _ = multi_tensor(
                        gtab, [crossing_evals1, crossing_evals2], S0=1.0,
                        angles=[(90, 0), (90, 60)],
                        fractions=[50, 50], snr=snr,
                    )
                    gt_tissue[x, y, z] = 2
                    eigs = 0.5 * crossing_evals1 + 0.5 * crossing_evals2
                else:
                    # CSF
                    signal, _ = multi_tensor(
                        gtab, [csf_evals], S0=1.0,
                        angles=[(0, 0)],
                        fractions=[100], snr=snr,
                    )
                    gt_tissue[x, y, z] = 3
                    eigs = csf_evals

                data[x, y, z, :] = signal
                eigs_si = eigs * 1e-6  # mm^2/s -> m^2/s
                fa_val, md_val = tensor_to_fa_md(jnp.array(eigs_si))
                gt_fa[x, y, z] = float(fa_val)
                gt_md[x, y, z] = float(md_val)

    mask = np.ones(vol_shape, dtype=np.uint8)
    ground_truth = {
        "FA": gt_fa,
        "MD": gt_md,
        "tissue": gt_tissue,
    }

    return data, bvals_fsl, bvecs, mask, ground_truth
