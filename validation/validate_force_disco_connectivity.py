#!/usr/bin/env python3
"""
DiSCo connectivity-matrix benchmark — FORCE paper §3.2 metric.

Per-SNR pipeline:
  1. Fit FORCE on DiSCo subject 1 single-shell b=1900 with the default
     library (the tuned library has the wm_threshold=1.0 ODF-suppression
     bug documented in doc 005 §4d).
  2. force_peaks(fit) → PeaksAndMetrics with peak directions.
  3. LocalTracking (Euler integration) on those peaks; seeds throughout
     the ROI mask.
  4. dipy.tracking.utils.connectivity_matrix → 16×16 connectivity.
  5. Pearson r on upper-triangle vs DiSCo ground-truth
     Connectivity_Matrix_Cross-Sectional_Area.

Reproduces (approximately) FORCE paper §3.2 reported numbers:
  r = 0.868 at SNR=10, r = 0.894 at SNR=50.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.validation.force_disco_connectivity import (
    connectivity_pearson,
    load_gt_connectivity,
    run_force_connectivity,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=int, nargs="+", default=[10, 30, 50])
    ap.add_argument("--seed-density", type=int, default=2)
    args = ap.parse_args()

    gt = load_gt_connectivity(subject=1)
    print(f"Ground-truth connectivity: 16×16, "
          f"{int((gt > 0).sum())} nonzero off-diagonal entries.")

    results = {}
    for snr in args.snrs:
        print(f"\n=== SNR = {snr} ===")
        res = run_force_connectivity(
            subject=1, snr=snr, tuned=False,
            seed_density=args.seed_density,
        )
        cmat = res["connectivity"]
        r = connectivity_pearson(cmat, gt)
        results[snr] = {
            "cmat": cmat,
            "r": r,
            "n_streamlines": res["streamlines_count"],
            "n_seeds": res["n_seeds"],
        }
        print(f"  n_seeds:        {res['n_seeds']:,}")
        print(f"  n_streamlines:  {res['streamlines_count']:,}")
        print(f"  Pearson r vs GT (upper-triangle): {r:.4f}")

    # Save NPZ
    out_npz = Path("validation/force_disco_connectivity_results.npz")
    np.savez(
        out_npz,
        snrs=np.array(args.snrs),
        r=np.array([results[s]["r"] for s in args.snrs]),
        gt=gt,
        **{f"cmat_snr{s}": results[s]["cmat"] for s in args.snrs},
    )
    print(f"\nWrote {out_npz}")

    # Figure: GT vs reconstructed matrices + Pearson curve
    n = len(args.snrs)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.5 * (n + 1), 6), dpi=130)
    # GT in first column (both rows the same)
    for row in range(2):
        ax = axes[row, 0]
        ax.imshow(gt, cmap="hot", vmin=0, vmax=gt.max())
        ax.set_title("Ground-truth\n(Cross-Sect Area)" if row == 0 else "")
        ax.set_xticks([]); ax.set_yticks([])

    for col, snr in enumerate(args.snrs):
        cmat = results[snr]["cmat"]
        r = results[snr]["r"]
        # Recon matrix
        ax = axes[0, col + 1]
        ax.imshow(cmat, cmap="hot", vmin=0, vmax=cmat.max() if cmat.max() > 0 else 1)
        ax.set_title(f"FORCE @ SNR={snr}\n(streamline counts)")
        ax.set_xticks([]); ax.set_yticks([])
        # Scatter GT vs recon
        ax = axes[1, col + 1]
        iu = np.triu_indices(16, k=1)
        x = gt[iu]
        y = cmat[iu]
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel("GT cross-sect area")
        ax.set_ylabel("FORCE streamline count")
        ax.set_title(f"r = {r:.3f}")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "DiSCo subject 1 connectivity matrix: FORCE peaks → LocalTracking "
        "→ ROI connectivity\n"
        "(default-library fit, single-shell b=1900, ROI-mask seeding, "
        "paper §3.2 reproduction)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = Path("validation/force_disco_connectivity.png")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # Print summary table
    print("\n" + "=" * 60)
    print("DiSCo connectivity-matrix Pearson r (upper-triangle)")
    print("=" * 60)
    print(f"  {'SNR':>4s}  {'r':>8s}  Paper §3.2 reported")
    paper_vals = {10: 0.868, 50: 0.894}
    for snr in args.snrs:
        paper = f"{paper_vals.get(snr, '—'):.3f}" if snr in paper_vals else "—"
        print(f"  {snr:>4d}  {results[snr]['r']:>8.4f}  {paper:>20s}")


if __name__ == "__main__":
    main()
