#!/usr/bin/env python3
"""Investigate the non-monotone failure modes of dipy's FORCE upstream
matcher observed in validate_force_replication_v2.py.

For each crossing angle, reports:

  1. Closest sphere-vertex angular error to mu1, mu2 (sphere quantisation
     limit — should be ~4 deg for the 362-vertex default_sphere).
  2. The matcher's reported num_fibers, dispersion, fracs on a fixed-seed
     noisy synthetic.
  3. The angular errors of the matcher's chosen fiber-direction labels.
  4. How many library entries are 2-fiber AND have one direction within
     10 deg of mu1 AND one within 10 deg of mu2 — the "good 2-fiber matches"
     available to the matcher for this crossing.

Run after validation/validate_force_replication_v2.py has cached its
500K-entry library at ~/.cache/dipy_force/force_v2_500k.npz.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import jax
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.reconst.force import FORCEModel, load_force_simulations


CACHE = Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"


def make_acq():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    v0 = np.array([[1.0, 0.0, 0.0]] * 2)
    v1 = np.asarray(jax.random.normal(k1, (32, 3)))
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = np.asarray(jax.random.normal(k2, (56, 3)))
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    bvals_si = np.concatenate([np.zeros(2), np.full(32, 1e9), np.full(56, 2e9)])
    bvecs = np.concatenate([v0, v1, v2], axis=0)
    return bvals_si, bvecs


def closest_vertex_err(mu, sphere=default_sphere):
    dots = np.abs(sphere.vertices @ mu)
    idx = int(np.argmax(dots))
    return idx, np.degrees(np.arccos(np.clip(dots[idx], -1.0, 1.0)))


def synthetic_2fiber_signal(bvals_si, bvecs, mu1, mu2, f1=0.45, f_iso=0.05,
                             d_par=1.7e-9, d_iso=3.0e-9):
    cos1 = bvecs @ mu1
    cos2 = bvecs @ mu2
    return (
        f1 * np.exp(-bvals_si * d_par * cos1 ** 2)
        + (1 - f1 - f_iso) * np.exp(-bvals_si * d_par * cos2 ** 2)
        + f_iso * np.exp(-bvals_si * d_iso)
    )


def add_rician_noise(signal, snr=30.0, seed=7):
    rng = np.random.default_rng(seed)
    sigma = 1.0 / snr
    n1 = rng.normal(0, sigma, signal.shape)
    n2 = rng.normal(0, sigma, signal.shape)
    return np.sqrt((signal + n1) ** 2 + n2 ** 2)


def main():
    if not CACHE.exists():
        raise SystemExit(f"FORCE library not found at {CACHE}; run "
                         "validate_force_replication_v2.py first.")

    bvals_si, bvecs = make_acq()
    gtab = gradient_table(bvals_si / 1e6, bvecs=bvecs)
    sims = load_force_simulations(str(CACHE))
    model = FORCEModel(gtab, simulations=sims, n_neighbors=50)

    labels = sims["labels"]
    num_fibers = sims["num_fibers"]
    n_total = labels.shape[0]
    print(f"Library size: {n_total:,}")
    print(f"Sphere: {default_sphere.vertices.shape[0]} vertices")
    print()
    print("num_fibers distribution:")
    for nf in np.unique(num_fibers):
        c = (num_fibers == nf).sum()
        print(f"  {int(nf)} fibers: {c:>9,} ({100*c/n_total:>4.1f}%)")
    print()

    angles = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    mu1 = np.array([0.0, 0.0, 1.0])
    print(
        f"{'ang':>3s}  {'vert-err':>10s}  {'num_fib':>7s}  "
        f"{'disp':>5s}  {'good 2-fib in lib':>18s}  label-dirs (errs to mu1, mu2)"
    )
    for ang in angles:
        rad = np.radians(ang)
        mu2 = np.array([np.sin(rad), 0.0, np.cos(rad)])

        # Sphere quantisation
        _, e1 = closest_vertex_err(mu1)
        _, e2 = closest_vertex_err(mu2)

        # Fit
        S = synthetic_2fiber_signal(bvals_si, bvecs, mu1, mu2)
        noisy = add_rician_noise(S).astype(np.float32)
        fit = model.fit(noisy[None, None, None, :])[0, 0, 0]
        label = np.asarray(fit.label)
        nz = np.where(label > 0)[0]

        # Library coverage: 2-fib entries with one direction near each truth
        cos1_v = np.abs(default_sphere.vertices @ mu1)
        cos2_v = np.abs(default_sphere.vertices @ mu2)
        near1 = np.where(cos1_v > np.cos(np.radians(10)))[0]
        near2 = np.where(cos2_v > np.cos(np.radians(10)))[0]
        is_2 = num_fibers == 2
        good = (
            is_2
            & (labels[:, near1] > 0).any(axis=1)
            & (labels[:, near2] > 0).any(axis=1)
        )
        n_good = int(good.sum())

        if len(nz):
            dirs_errs = []
            for idx in nz[:3]:
                d = default_sphere.vertices[idx]
                e_mu1 = np.degrees(np.arccos(np.clip(abs(d @ mu1), -1, 1)))
                e_mu2 = np.degrees(np.arccos(np.clip(abs(d @ mu2), -1, 1)))
                dirs_errs.append(f"({e_mu1:.0f},{e_mu2:.0f})")
            dirs_str = " ".join(dirs_errs)
        else:
            dirs_str = "<no labels>"

        print(
            f"{ang:>3d}  ({e1:>3.1f},{e2:>3.1f})  {fit.num_fibers:>6.0f}  "
            f"{fit.dispersion:>5.3f}  {n_good:>10,} ({100*n_good/n_total:>4.2f}%)  "
            f"{dirs_str}"
        )


if __name__ == "__main__":
    main()
