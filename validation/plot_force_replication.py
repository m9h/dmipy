#!/usr/bin/env python3
"""Plot FORCE replication results.

Auto-detects v2 (9-method) results if present, falls back to v1 (3-method).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


V2_METHODS = [
    ("dict", "dmipy-JAX dictionary", "o-", "#1f77b4"),
    ("hybrid_guard", "Hybrid (MSE-guarded)", "s-", "#2ca02c"),
    ("hybrid15", "Hybrid (LM, maxiter=15)", "v-", "#9467bd"),
    ("hybrid50", "Hybrid (LM, maxiter=50)", "^-", "#8c564b"),
    ("lm", "Pure LM (random init)", "x--", "#e377c2"),
    ("dipy_force", "DIPY FORCE (force_peaks)", "D-", "#d62728"),
    ("dipy_force_internal", "DIPY FORCE-internal (label)", "P:", "#ff7f0e"),
    ("csd", "DIPY CSD (peaks)", "+-", "#17becf"),
    ("gqi", "DIPY GQI (peaks)", "*-", "#7f7f7f"),
]

V1_METHODS = [
    ("dict_rates", "Dictionary (FORCE-style)", "o-", "#1f77b4"),
    ("hybrid_rates", "Dictionary + LM (hybrid)", "s-", "#2ca02c"),
    ("lm_rates", "Pure LM (random init)", "^--", "#e377c2"),
]

THREE_FIBER_METHODS = [
    ("dict3", "dmipy-JAX 3-stick dict (200K)", "o-", "#1f77b4"),
    ("dipy_force", "DIPY FORCE (force_peaks)", "D-", "#d62728"),
    ("dipy_force_internal", "DIPY FORCE-internal (label)", "P:", "#ff7f0e"),
    ("csd", "DIPY CSD (peaks)", "+-", "#17becf"),
    ("gqi", "DIPY GQI (peaks)", "*-", "#7f7f7f"),
]


SNR_SWEEP_METHODS = [
    ("dict",       "dmipy-JAX dictionary"),
    ("dipy_force", "DIPY FORCE (force_peaks)"),
    ("csd",        "DIPY CSD (peaks)"),
    ("gqi",        "DIPY GQI (peaks)"),
]


def plot_snr_sweep(npz_path: Path, out: Path, title_suffix: str = None):
    r = np.load(npz_path)
    snrs = r["snrs"]
    angles = r["angles"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=140, sharex=True, sharey=True)
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")
    snr_colors = [cmap(0.15 + 0.7 * i / max(len(snrs) - 1, 1)) for i in range(len(snrs))]

    for ax, (key, label) in zip(axes, SNR_SWEEP_METHODS):
        for s_idx, snr in enumerate(snrs):
            ax.plot(angles, r[key][s_idx] * 100, "o-", lw=1.6, ms=4,
                    color=snr_colors[s_idx], label=f"SNR = {int(snr)}")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-5, 105)
        ax.set_xticks(angles[::2])
        ax.grid(alpha=0.3)
        ax.axvspan(10, 40, alpha=0.06, color="red")

    for ax in axes[2:]:
        ax.set_xlabel("Crossing angle (deg)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Both-fibres detection rate (%)")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.suptitle(
        title_suffix or (
            "FORCE SNR sweep: 4 tools × 4 SNR levels × 17 crossing angles "
            "(2-stick, 200 trials/cell)"
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="Path to .npz; auto-detects snr-sweep -> 3fiber -> v2 -> v1.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    matched = Path("validation/force_matched_results.npz")
    snr = Path("validation/force_snr_sweep_results.npz")
    if (args.input is None and matched.exists()) or (
        args.input is not None and "matched" in str(args.input)
    ):
        npz_path = args.input or matched
        out = args.output or Path("validation/force_matched.png")
        return plot_snr_sweep(npz_path, out, title_suffix=(
            "FORCE-paper-matched re-run: Stanford HARDI 150 dirs × b=2000, "
            "Bingham-dispersed sticks (ODI∈[0.01,0.30]), 20° tolerance"
        ))
    if (args.input is None and snr.exists()) or (
        args.input is not None and "snr_sweep" in str(args.input)
    ):
        npz_path = args.input or snr
        out = args.output or Path("validation/force_snr_sweep.png")
        return plot_snr_sweep(npz_path, out)

    v3f = Path("validation/force_3fiber_results.npz")
    v2 = Path("validation/force_replication_v2_results.npz")
    v1 = Path("validation/force_replication_results.npz")

    if args.input is not None:
        npz_path = args.input
    elif v3f.exists():
        npz_path = v3f
    elif v2.exists():
        npz_path = v2
    elif v1.exists():
        npz_path = v1
    else:
        raise SystemExit("No results .npz found in validation/")

    files = set(np.load(npz_path).files)
    is_3fiber = "dict3" in files
    is_v2 = "dipy_force" in files and not is_3fiber
    if is_3fiber:
        methods = THREE_FIBER_METHODS
    elif is_v2:
        methods = V2_METHODS
    else:
        methods = V1_METHODS
    out = args.output or Path(
        "validation/force_3fiber.png" if is_3fiber else
        "validation/force_replication_v2.png" if is_v2 else
        "validation/force_replication.png"
    )

    r = np.load(npz_path)
    angles = r["alpha"] if is_3fiber else r["angles"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for key, label, style, color in methods:
        if key not in r.files:
            continue
        ax.plot(angles, r[key] * 100, style, lw=1.6, ms=5,
                label=label, color=color, alpha=0.9)

    if is_3fiber:
        ax.set_xlabel("Half-spread α (deg) — three sticks at θ ∈ {−α, 0, +α}")
        ax.set_ylabel("All-3-fibres detection rate (%)")
        title = ("FORCE 3-fiber benchmark: coplanar 3-stick (+x/+z), SNR 30, "
                 "200 trials/α\n(dipy 1.12.1 FORCEModel @ 500K library, "
                 "70% 3-fiber configs)")
    else:
        ax.axvspan(10, 40, alpha=0.07, color="red",
                   label="Shallow-crossing regime (FORCE paper focus)")
        ax.set_xlabel("Crossing angle (deg)")
        ax.set_ylabel("Both-fibers detection rate (%)")
        title = "FORCE Replication v2: 2-fiber crossing, SNR 30, 200 trials/angle"
        if is_v2:
            title += "\n(dipy 1.12.1 FORCEModel @ 500K library, n_neighbors=50)"
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_xticks(angles)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
