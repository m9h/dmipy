#!/usr/bin/env python3
"""4-method DiSCo connectivity comparison figure."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_r(path: Path, snrs: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    d = np.load(path)
    if (np.asarray(d["snrs"]) == snrs).all():
        return np.asarray(d["r"])
    return None


def main():
    snrs = np.array([10, 30, 50])
    dmipy_stick    = np.array([0.7611, 0.7905, 0.8228])
    dmipy_stick_ms = _load_r(
        Path("validation/dmipy_disco_connectivity_results_multishell.npz"),
        snrs,
    )
    dmipy_zep = _load_r(
        Path("validation/dmipy_disco_connectivity_zeppelin_results.npz"),
        snrs,
    )
    dmipy_zep_ms = _load_r(
        Path("validation/dmipy_disco_connectivity_zeppelin_results_multishell.npz"),
        snrs,
    )
    force   = np.array([0.298, 0.211, 0.322])
    mrtrix  = np.array([0.142, 0.081, 0.131])
    paper_snrs = np.array([10, 50])
    paper_r    = np.array([0.868, 0.894])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    ax.plot(snrs, dmipy_stick, "o-", lw=2, ms=9,
            label="dmipy-JAX 3D stick, single-shell (§21)",
            color="#1f77b4", alpha=0.5)
    if dmipy_stick_ms is not None:
        ax.plot(snrs, dmipy_stick_ms, "o-", lw=2.5, ms=10,
                label="dmipy-JAX 3D stick, multi-shell (§24)",
                color="#1f77b4")
    if dmipy_zep is not None:
        ax.plot(snrs, dmipy_zep, "v-", lw=2, ms=9,
                label="dmipy-JAX stick+zeppelin, single-shell (§22)",
                color="#17becf", alpha=0.5)
    if dmipy_zep_ms is not None:
        ax.plot(snrs, dmipy_zep_ms, "v-", lw=2.5, ms=10,
                label="dmipy-JAX stick+zeppelin, multi-shell (§24)",
                color="#17becf")
    ax.plot(snrs, force, "s-", lw=2, ms=8, label="dipy FORCE (§17.5)",
            color="#d62728")
    ax.plot(snrs, mrtrix, "^-", lw=2, ms=8, label="MRtrix SD_STREAM (§18)",
            color="#9467bd")
    ax.plot(paper_snrs, paper_r, "D--", lw=2, ms=10,
            label="FORCE paper §3.2 reported",
            color="#2ca02c", alpha=0.85)

    ax.set_xlabel("Rician SNR")
    ax.set_ylabel("Connectivity-matrix Pearson r vs DiSCo GT (upper triangle)")
    ax.set_title(
        "DiSCo connectivity reproduction: 4-method comparison\n"
        "(Subject 1, single-shell b=1900, identical seeds + stopping criterion)",
        fontsize=11,
    )
    ax.set_xticks([10, 30, 50])
    ax.set_xlim(5, 55)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    out = Path("validation/disco_connectivity_4method_comparison.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
