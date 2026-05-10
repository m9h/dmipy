#!/usr/bin/env python3
"""
DiSCo phantom benchmark — FORCE paper §3.2 protocol.

Single-shell b≈2000 extraction (DiSCo's b=1900) on subject 1 highRes
(40³ voxels). Three methods fit:
  - FORCE (dipy 1.12.1, 500K-entry library generated for the DiSCo gtab)
  - DTI (dipy TensorModel) — sanity baseline
  - CSD (dipy ConstrainedSphericalDeconvModel) — paper's stated baseline

Ground-truth comparisons (brain-mask, Pearson r against
``highRes_DiSCo1_Strand_Intra_Volume_Fraction``):
  - FORCE NDI (``fit.nd``)
  - FORCE FA (vs DTI FA round-trip — sanity)

Output:
  - validation/force_disco.png — 4-panel: GT NDI, FORCE NDI, FORCE FA,
    scatter GT vs FORCE NDI
  - Pearson correlations printed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.validation.force_disco import load_disco_subject
from dmipy_jax.validation.force_inter_method import (
    fit_dti_baseline,
    masked_pearson,
)


DIPY_FORCE_DISCO_CACHE = Path.home() / ".cache" / "dipy_force" / "force_disco_500k.npz"
DISCO_FORCE_FIT_CACHE = Path.home() / ".cache" / "force_disco" / "force_disco_fit.npz"


def get_or_make_force_library(gtab, n_sims=500_000):
    from dipy.reconst.force import (
        generate_force_simulations,
        load_force_simulations,
        save_force_simulations,
    )

    DIPY_FORCE_DISCO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DIPY_FORCE_DISCO_CACHE.exists():
        print(f"Loading cached DiSCo FORCE library from {DIPY_FORCE_DISCO_CACHE}")
        return load_force_simulations(str(DIPY_FORCE_DISCO_CACHE))
    print(f"Generating DiSCo FORCE library (n={n_sims:,}, single-CPU, ~15 min)...")
    sims = generate_force_simulations(
        gtab, num_simulations=n_sims, num_cpus=1, verbose=True,
    )
    save_force_simulations(sims, str(DIPY_FORCE_DISCO_CACHE))
    print(f"Saved to {DIPY_FORCE_DISCO_CACHE}")
    return sims


def fit_or_load_force_disco(data, gtab, mask, sims, n_neighbors=50, n_jobs=-1):
    import warnings as _w
    _w.filterwarnings("ignore")
    from dipy.reconst.force import FORCEModel

    if DISCO_FORCE_FIT_CACHE.exists():
        print(f"Loading cached FORCE fit from {DISCO_FORCE_FIT_CACHE}")
        d = np.load(DISCO_FORCE_FIT_CACHE)
        return {k: d[k] for k in d.files}

    model = FORCEModel(gtab, simulations=sims, n_neighbors=n_neighbors)
    print(f"Fitting FORCE on {int(mask.sum()):,} masked voxels (this takes a few minutes)...")
    fit = model.fit(data, mask=mask, n_jobs=n_jobs)

    out = {
        "fa": np.asarray(fit.fa, dtype=np.float32),
        "md": np.asarray(fit.md, dtype=np.float32),
        "rd": np.asarray(fit.rd, dtype=np.float32),
        "nd": np.asarray(fit.nd, dtype=np.float32),
        "dispersion": np.asarray(fit.dispersion, dtype=np.float32),
        "wm_fraction": np.asarray(fit.wm_fraction, dtype=np.float32),
        "num_fibers": np.asarray(fit.num_fibers, dtype=np.float32),
    }
    DISCO_FORCE_FIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DISCO_FORCE_FIT_CACHE, **out)
    return out


def render_disco_figure(force_maps, dti_maps, gt, mask, out_png):
    z = int(np.median(np.argwhere(mask.any(axis=(0, 1)))[:, 0]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=130)

    panels = [
        (axes[0, 0], gt["ndi"], "Ground-truth NDI (Intra VF)", "viridis", 0, 1),
        (axes[0, 1], force_maps["nd"], "FORCE NDI (fit.nd)", "viridis", 0, 1),
        (axes[0, 2], np.abs(force_maps["nd"] - gt["ndi"]) * mask,
         "|FORCE − GT| NDI", "hot", 0, 0.5),
        (axes[1, 0], dti_maps["fa"], "DTI FA", "gray", 0, 1),
        (axes[1, 1], force_maps["fa"], "FORCE FA", "gray", 0, 1),
        (axes[1, 2], force_maps["dispersion"], "FORCE ODI / dispersion",
         "viridis", 0, 1),
    ]
    for ax, arr, title, cmap, vmin, vmax in panels:
        im = ax.imshow(np.rot90(arr[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "DiSCo phantom benchmark (subject 1 highRes, single-shell b=1900, SNR=30)\n"
        "FORCE vs ground-truth Intra Volume Fraction; DTI baseline",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")


def main():
    print("=" * 78)
    print("DiSCo phantom benchmark — FORCE paper §3.2 protocol")
    print("=" * 78)

    print("\nLoading DiSCo subject 1 (single-shell b=1900, SNR=30)...")
    out = load_disco_subject(subject=1, snr=30, single_shell_b=1900)
    data = out["data"]
    mask = out["mask"]
    gtab = out["gtab"]
    gt = out["gt"]
    print(f"  data: {data.shape}, mask: {int(mask.sum()):,} voxels")

    print("\nGetting FORCE library (regenerates if not cached)...")
    sims = get_or_make_force_library(gtab, n_sims=500_000)

    print("\nFitting FORCE...")
    force_maps = fit_or_load_force_disco(data, gtab, mask, sims)
    for k in ("fa", "nd", "dispersion"):
        v = force_maps[k][mask]
        print(f"  FORCE {k:>4s}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    print("\nFitting DTI baseline...")
    dti_maps = fit_dti_baseline(data, gtab, mask)
    for k in ("fa", "md"):
        v = dti_maps[k][mask]
        print(f"  DTI {k:>3s}: min={v.min():.4f}  mean={v.mean():.4f}  max={v.max():.4f}")

    # Ground-truth Pearson correlations
    print("\n" + "=" * 78)
    print("Brain-mask Pearson correlations vs ground truth")
    print("=" * 78)
    r_ndi = masked_pearson(force_maps["nd"], gt["ndi"], mask)
    r_force_dti_fa = masked_pearson(force_maps["fa"], dti_maps["fa"], mask)
    print(f"  FORCE NDI vs GT Intra Volume Fraction: r = {r_ndi:.4f}")
    print(f"  FORCE FA  vs DTI FA (sanity round-trip): r = {r_force_dti_fa:.4f}")

    out_png = Path("validation/force_disco.png")
    render_disco_figure(force_maps, dti_maps, gt, mask, out_png)

    # Save the result npz for later
    out_npz = Path("validation/force_disco_results.npz")
    np.savez(
        out_npz,
        force_nd=force_maps["nd"],
        force_fa=force_maps["fa"],
        force_dispersion=force_maps["dispersion"],
        dti_fa=dti_maps["fa"],
        dti_md=dti_maps["md"],
        gt_ndi=gt["ndi"],
        gt_diameter=gt["diameter"],
        mask=mask,
        r_ndi=r_ndi,
        r_force_dti_fa=r_force_dti_fa,
    )
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
