#!/usr/bin/env python3
"""Render the DiSCo multi-SNR result table + figure.

Reads validation/force_disco_results_tuned_snr{10,20,50}.npz and
validation/force_disco_results_tuned.npz (SNR=30 from §16.3), computes
brain-mask Pearson correlations of FORCE NDI vs ground truth, then
plots NDI + FA recovery as a function of SNR.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    sources = {
        10: Path("validation/force_disco_results_tuned_snr10.npz"),
        20: Path("validation/force_disco_results_tuned_snr20.npz"),
        50: Path("validation/force_disco_results_tuned_snr50.npz"),
    }

    # SNR=30 baseline from doc 004 §16.3 (commit a2b1d94); the original NPZ
    # was overwritten by the multi-SNR runner's rename step. Numbers are
    # permanent in git history.
    snrs = [30]
    r_ndi = [0.9180]
    r_fa = [0.9902]
    for snr in sorted(sources):
        p = sources[snr]
        if not p.exists():
            print(f"  SNR={snr}: {p} missing")
            continue
        d = np.load(p)
        snrs.append(snr)
        r_ndi.append(float(d["r_ndi"]))
        r_fa.append(float(d["r_force_dti_fa"]))

    # Sort by SNR for monotone plot
    order = np.argsort(snrs)
    snrs = np.asarray(snrs)[order]
    r_ndi = np.asarray(r_ndi)[order]
    r_fa = np.asarray(r_fa)[order]

    print("\nBrain-mask Pearson correlations (tuned-library DiSCo):")
    print(f"  {'SNR':>4s}   {'NDI':>6s}   {'FA':>6s}")
    for snr, n, f in zip(snrs, r_ndi, r_fa):
        print(f"  {int(snr):>4d}   {n:>6.4f}   {f:>6.4f}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.plot(snrs, r_ndi, "o-", lw=2, ms=8, label="FORCE NDI vs GT Intra Vol Fraction",
            color="#1f77b4")
    ax.plot(snrs, r_fa, "s-", lw=2, ms=8, label="FORCE FA vs DTI FA",
            color="#2ca02c")

    # Paper §3.2 reported connectivity-matrix correlations (different metric, ballpark)
    paper_snrs = np.array([10, 50])
    paper_r = np.array([0.868, 0.894])
    ax.plot(paper_snrs, paper_r, "D--", lw=1.5, ms=7, alpha=0.7,
            color="#d62728",
            label="Paper §3.2 reported (connectivity-matrix, ≠ same metric)")

    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50])
    ax.set_xticklabels(["10", "20", "30", "50"])
    ax.set_xlim(8, 65)
    ax.set_ylim(0.83, 1.005)
    ax.set_xlabel("Rician SNR")
    ax.set_ylabel("Brain-mask Pearson r")
    ax.set_title(
        "DiSCo phantom benchmark (subject 1 highRes, single-shell b=1900,\n"
        "FORCE library tuned per paper §3.2; 15,267 voxels per cell)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    out = Path("validation/force_disco_multi_snr.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
