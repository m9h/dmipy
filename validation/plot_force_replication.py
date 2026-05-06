#!/usr/bin/env python3
"""Plot FORCE replication results: peak detection rate vs crossing angle."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    npz_path = Path("validation/force_replication_results.npz")
    if not npz_path.exists():
        raise SystemExit(f"Results not found at {npz_path}; run validate_force_replication.py first.")

    r = np.load(npz_path)
    angles = r["angles"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
    ax.plot(angles, r["dict_rates"] * 100, "o-", lw=2, ms=6, label="Dictionary (FORCE-style)")
    ax.plot(angles, r["hybrid_rates"] * 100, "s-", lw=2, ms=6, label="Dictionary + LM (hybrid)")
    ax.plot(angles, r["lm_rates"] * 100, "^--", lw=2, ms=6, label="Pure LM (random init)")

    ax.axvspan(10, 40, alpha=0.08, color="red", label="Shallow-crossing regime")
    ax.set_xlabel("Crossing angle (deg)")
    ax.set_ylabel("Both-fibers detection rate (%)")
    ax.set_title("FORCE Replication: 2-fiber crossing, SNR 30, 200 trials/angle")
    ax.set_ylim(-5, 105)
    ax.set_xticks(angles)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out = Path("validation/force_replication.png")
    fig.tight_layout()
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
