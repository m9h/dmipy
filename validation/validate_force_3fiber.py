#!/usr/bin/env python3
"""
Companion to validate_force_replication_v2.py: 3-fiber crossing benchmark.

dipy's FORCE library is ~70% 3-fiber configurations (Dirichlet(2,1,1)
prior over fibre fractions); 2-fiber benchmarks understate its capability.
This script tests whether FORCE *excels* in the 3-fiber regime its library
was designed for, and how a freshly-built dmipy-JAX 3-stick library
compares.

Setup
-----
- Synthetic: 3 sticks coplanar in +x/+z, equal fractions (0.317 each),
  isotropic FW = 0.05, d_par = 1.7e-9, d_iso = 3e-9.
- Geometry sweep: vary the *minimum inter-fibre angle*. The three sticks
  are placed at theta = -alpha, 0, +alpha relative to z (alpha controls
  spread). At alpha=15 deg the three sticks span -15 to +15 (min sep 15).
  At alpha=60 deg they span -60 to +60 (min sep 60).
- Acquisition: same 90-direction 2-shell as v2.
- SNR 30, Rician noise, 200 trials per geometry.
- Detection criterion: all 3 peaks within 15 deg of their assigned truth.

Methods
-------
- dmipy-JAX 3-stick dictionary (fresh 200K-entry library)
- DIPY upstream FORCEModel (reuses 500K cached library from v2)
- DIPY FORCE-internal (label-based, bypasses force_peaks)
- DIPY CSD peaks
- DIPY GQI peaks

Cost: ~30-40 sec library build + ~2-2.5 h sweep on GB10. Use
``--smoke`` to do a single 10-trial check at one alpha.
"""

import argparse
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.force import FORCEModel, force_peaks, load_force_simulations
from dipy.reconst.gqi import GeneralizedQSamplingModel

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.validation.force_helpers import (
    acq_to_gtab,
    best_three_peaks,
    check_all_three_detected,
    params3_to_orientations,
)


DIPY_FORCE_CACHE = Path(os.environ.get(
    "DIPY_FORCE_CACHE",
    str(Path.home() / ".cache" / "dipy_force" / "force_v2_500k.npz"),
))


# --------------------------------------------------------------------------- #
# Acquisition (matches v2 exactly so we can reuse its dipy-FORCE library)
# --------------------------------------------------------------------------- #

def make_multishell_acquisition():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.array([[1.0, 0.0, 0.0]] * 2)
    v1 = rand_vecs(k1, 32)
    v2 = rand_vecs(k2, 56)
    bvals = jnp.concatenate([jnp.zeros(2), jnp.full(32, 1e9), jnp.full(56, 2e9)])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


# --------------------------------------------------------------------------- #
# 3-fibre forward model and ground-truth synthesiser
# --------------------------------------------------------------------------- #

def three_stick_signal(acq, mu1, mu2, mu3, f1, f2, f_iso=0.05,
                       d_par=1.7e-9, d_iso=3.0e-9):
    f3 = 1.0 - f1 - f2 - f_iso
    cos1 = acq.gradient_directions @ mu1
    cos2 = acq.gradient_directions @ mu2
    cos3 = acq.gradient_directions @ mu3
    s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
    s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
    s3 = jnp.exp(-acq.bvalues * d_par * cos3 ** 2)
    s_iso = jnp.exp(-acq.bvalues * d_iso)
    return f1 * s1 + f2 * s2 + f3 * s3 + f_iso * s_iso


def build_three_stick_simulator(acq):
    """7-param planar 3-stick simulator for library generation.

    params: [d_par, theta1, theta2, theta3, f1, f2, f_iso]
    """
    def forward_fn(params, acq):
        d_par = params[0]
        t1, t2, t3 = params[1], params[2], params[3]
        f1, f2, f_iso = params[4], params[5], params[6]
        f3 = 1.0 - f1 - f2 - f_iso
        mu1 = jnp.array([jnp.sin(t1), 0.0, jnp.cos(t1)])
        mu2 = jnp.array([jnp.sin(t2), 0.0, jnp.cos(t2)])
        mu3 = jnp.array([jnp.sin(t3), 0.0, jnp.cos(t3)])
        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2
        cos3 = acq.gradient_directions @ mu3
        s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
        s3 = jnp.exp(-acq.bvalues * d_par * cos3 ** 2)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f3 * s3 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "theta3", "f1", "f2", "f_iso"],
        parameter_ranges={
            "d_par": (1.0e-9, 2.5e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "theta3": (0.0, float(jnp.pi)),
            "f1": (0.1, 0.6),
            "f2": (0.1, 0.6),
            "f_iso": (0.0, 0.2),
        },
        acquisition=acq,
    )


# --------------------------------------------------------------------------- #
# DIPY peak helpers (delegate to peaks_from_model)
# --------------------------------------------------------------------------- #

def csd_peaks_3(noisy_signal_np, gtab, response, sphere):
    data4d = noisy_signal_np[None, None, None, :]
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
    peaks = peaks_from_model(
        model=csd_model, data=data4d, sphere=sphere,
        relative_peak_threshold=0.3,
        min_separation_angle=10.0,
        return_odf=False, normalize_peaks=True, npeaks=5,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def gqi_peaks_3(noisy_signal_np, gtab, sphere):
    data4d = noisy_signal_np[None, None, None, :]
    gqi_model = GeneralizedQSamplingModel(gtab, sampling_length=1.2)
    peaks = peaks_from_model(
        model=gqi_model, data=data4d, sphere=sphere,
        relative_peak_threshold=0.3,
        min_separation_angle=10.0,
        return_odf=False, normalize_peaks=True, npeaks=5,
    )
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def dipy_force_peaks_3(noisy_signal_np, force_model):
    data4d = noisy_signal_np[None, None, None, :].astype(np.float32)
    fit = force_model.fit(data4d)
    peaks = force_peaks(fit)
    return [peaks.peak_dirs[0, 0, 0, k] for k in range(peaks.peak_dirs.shape[-2])]


def dipy_force_internal_peaks_3(noisy_signal_np, force_model, sphere):
    data4d = noisy_signal_np[None, None, None, :].astype(np.float32)
    fit = force_model.fit(data4d)[0, 0, 0]
    label = np.asarray(fit.label)
    nz = np.where(label > 0)[0]
    if nz.size == 0:
        return []
    return [sphere.vertices[i] for i in nz]


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run a single alpha=30 deg, 10-trial smoke test")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--lib-size", type=int, default=200_000,
                    help="dmipy-JAX 3-stick library size")
    args = ap.parse_args()

    print("=" * 78)
    print("FORCE 3-fiber benchmark: dipy FORCE vs dmipy-JAX 3-stick dict + ODF baselines")
    print("=" * 78)

    acq = make_multishell_acquisition()
    sim = build_three_stick_simulator(acq)

    print(f"\nGenerating dmipy-JAX 3-stick library ({args.lib_size:,} entries)...")
    gen = LibraryGenerator(sim, chunk_size=50_000)
    lib_params, lib_signals = gen.generate(args.lib_size, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    lib = SimulationLibrary(
        params=lib_params, signals=lib_signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)
    print(f"Library: {lib.n_entries:,} entries, signal dim {lib.signal_dim}")

    gtab = acq_to_gtab(acq)
    response = (np.array([1.7e-3, 3e-4, 3e-4]), 1.0)
    sphere = default_sphere

    if not DIPY_FORCE_CACHE.exists():
        raise SystemExit(
            f"DIPY FORCE library cache not found at {DIPY_FORCE_CACHE}. "
            "Run validate_force_replication_v2.py first to build it."
        )
    dipy_sims = load_force_simulations(str(DIPY_FORCE_CACHE))
    dipy_force = FORCEModel(gtab, simulations=dipy_sims, n_neighbors=50)
    print("DIPY FORCEModel ready (500K sims, n_neighbors=50).")

    # Geometry: alpha controls fiber spread. Three sticks at theta = -alpha, 0, +alpha.
    if args.smoke:
        alphas = [30]
        n_trials = 10
    else:
        alphas = list(range(15, 65, 5))   # 15..60 deg fiber spread
        n_trials = args.n_trials

    method_names = ["dict3", "dipy_force", "dipy_force_internal", "csd", "gqi"]
    results = {m: [] for m in method_names}
    results["alpha"] = []
    snr = 30.0
    angle_threshold = 15.0
    f1_true = 0.317
    f2_true = 0.317
    f_iso_true = 0.05

    key = jax.random.PRNGKey(42)

    for alpha in alphas:
        rad = np.radians(alpha)
        mu1_true = np.array([-np.sin(rad), 0.0, np.cos(rad)])
        mu2_true = np.array([0.0, 0.0, 1.0])
        mu3_true = np.array([np.sin(rad), 0.0, np.cos(rad)])
        counts = {m: 0 for m in method_names}

        for trial in range(n_trials):
            key, k_noise = jax.random.split(key, 2)
            signal = three_stick_signal(
                acq,
                jnp.asarray(mu1_true), jnp.asarray(mu2_true), jnp.asarray(mu3_true),
                f1_true, f2_true, f_iso_true,
            )
            sigma = 1.0 / snr
            k1, k2 = jax.random.split(k_noise)
            n1 = jax.random.normal(k1, signal.shape) * sigma
            n2 = jax.random.normal(k2, signal.shape) * sigma
            noisy = jnp.sqrt((signal + n1) ** 2 + n2 ** 2)
            noisy_np = np.asarray(noisy)

            # dmipy-JAX 3-stick dictionary
            dict3_params, _ = matcher.match_single(noisy)
            mu1_d, mu2_d, mu3_d = params3_to_orientations(np.asarray(dict3_params))
            if check_all_three_detected(
                mu1_true, mu2_true, mu3_true, mu1_d, mu2_d, mu3_d, angle_threshold,
            ):
                counts["dict3"] += 1

            # dipy upstream FORCE
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = dipy_force_peaks_3(noisy_np, dipy_force)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["dipy_force"] += 1
            except Exception:
                pass

            # dipy FORCE-internal (read FORCEFit.label directly)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = dipy_force_internal_peaks_3(noisy_np, dipy_force, sphere)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["dipy_force_internal"] += 1
            except Exception:
                pass

            # CSD peaks
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = csd_peaks_3(noisy_np, gtab, response, sphere)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["csd"] += 1
            except Exception:
                pass

            # GQI peaks
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pks = gqi_peaks_3(noisy_np, gtab, sphere)
                triple = best_three_peaks(pks, mu1_true, mu2_true, mu3_true)
                if triple is not None and check_all_three_detected(
                    mu1_true, mu2_true, mu3_true,
                    triple[0], triple[1], triple[2], angle_threshold,
                ):
                    counts["gqi"] += 1
            except Exception:
                pass

        results["alpha"].append(alpha)
        for m in method_names:
            results[m].append(counts[m] / n_trials)

        line = f"  alpha={alpha:3d} deg:" + "".join(
            f"  {m}={results[m][-1]:.0%}" for m in method_names
        )
        print(line)

    print("\n" + "=" * 78)
    print("Summary: all-3-fibres detection rate by fiber spread (alpha = half-spread)")
    print("=" * 78)
    header = f"{'alpha':>5s}  " + "  ".join(f"{m:>16s}" for m in method_names)
    print(header)
    for i, a in enumerate(results["alpha"]):
        row = f"{a:>5d}  " + "  ".join(
            f"{results[m][i]:>15.1%}" for m in method_names
        )
        print(row)

    out = Path("validation/force_3fiber_results.npz")
    if not args.smoke:
        np.savez(
            out,
            alpha=np.asarray(results["alpha"]),
            **{m: np.asarray(results[m]) for m in method_names},
        )
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
