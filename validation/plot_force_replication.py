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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="Path to .npz; auto-detects v2 then v1.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

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
