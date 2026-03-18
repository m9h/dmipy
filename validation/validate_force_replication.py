#!/usr/bin/env python3
"""
Validation 1: FORCE replication — crossing-angle peak detection.

Generates synthetic 2-fiber data at crossing angles 10-90 degrees and
compares peak detection rates for:
  - Dictionary matching (FORCE-style)
  - Dictionary + LM hybrid (dict init → LM refinement)
  - Pure LM (random init → LM)

Replicates the key FORCE result: dictionary matching outperforms at
shallow crossings (10-40 degrees) where ODF-based methods fail.

Usage::

    python validation/validate_force_replication.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary
from dmipy_jax.library.matcher import DictionaryMatcher
from dmipy_jax.fitting.optimization import VoxelFitter


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


def two_stick_signal(acq, mu1, mu2, f1, d_par=1.7e-9, d_iso=3.0e-9, f_iso=0.05):
    """Generate a 2-fiber signal with given orientations."""
    f2 = 1.0 - f1 - f_iso
    cos1 = acq.gradient_directions @ mu1
    cos2 = acq.gradient_directions @ mu2
    s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
    s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
    s_iso = jnp.exp(-acq.bvalues * d_iso)
    return f1 * s1 + f2 * s2 + f_iso * s_iso


def build_library_simulator(acq):
    """2-stick simulator for library generation."""
    def forward_fn(params, acq):
        d_par = params[0]
        theta1 = params[1]
        theta2 = params[2]
        f1 = params[3]
        f_iso = params[4]
        f2 = 1.0 - f1 - f_iso

        mu1 = jnp.array([jnp.sin(theta1), 0.0, jnp.cos(theta1)])
        mu2 = jnp.array([jnp.sin(theta2), 0.0, jnp.cos(theta2)])

        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2
        s1 = jnp.exp(-acq.bvalues * d_par * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_par * cos2 ** 2)
        s_iso = jnp.exp(-acq.bvalues * 3.0e-9)
        return f1 * s1 + f2 * s2 + f_iso * s_iso

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_par", "theta1", "theta2", "f1", "f_iso"],
        parameter_ranges={
            "d_par": (1.0e-9, 2.5e-9),
            "theta1": (0.0, float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)),
            "f1": (0.1, 0.8),
            "f_iso": (0.0, 0.2),
        },
        acquisition=acq,
    )


def angular_error(mu_true, mu_recovered):
    """Minimum angle between two unit vectors (handles antipodal symmetry)."""
    dot = np.abs(np.dot(mu_true, mu_recovered))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def params_to_orientations(params_np):
    """Extract mu1, mu2 from flat params [d_par, theta1, theta2, f1, f_iso]."""
    theta1, theta2 = params_np[1], params_np[2]
    mu1 = np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    mu2 = np.array([np.sin(theta2), 0.0, np.cos(theta2)])
    return mu1, mu2


def check_both_detected(mu1_true, mu2_true, mu1_rec, mu2_rec, threshold):
    """Check if both fibers are detected within angular threshold."""
    err_11 = angular_error(mu1_true, mu1_rec)
    err_12 = angular_error(mu1_true, mu2_rec)
    err_21 = angular_error(mu2_true, mu1_rec)
    err_22 = angular_error(mu2_true, mu2_rec)
    if err_11 + err_22 < err_12 + err_21:
        return err_11 < threshold and err_22 < threshold
    else:
        return err_12 < threshold and err_21 < threshold


def build_lm_fitter(forward_fn, acq):
    """Build a VoxelFitter (L-BFGS-B) for the 2-stick model."""
    param_ranges = [
        (1.0e-9, 2.5e-9),   # d_par
        (0.0, float(jnp.pi)),  # theta1
        (0.0, float(jnp.pi)),  # theta2
        (0.1, 0.8),          # f1
        (0.0, 0.2),          # f_iso
    ]
    scales = jnp.array([1e-9, 1.0, 1.0, 1.0, 1.0])
    return VoxelFitter(forward_fn, param_ranges, scales=scales,
                       solver_settings={"maxiter": 50, "tol": 1e-6})


def main():
    print("=" * 70)
    print("FORCE Replication: Crossing-Angle Peak Detection")
    print("  Comparing: Dictionary / Dictionary+LM Hybrid / Pure LM")
    print("=" * 70)

    acq = make_multishell_acquisition()
    sim = build_library_simulator(acq)

    # Generate library
    print("\nGenerating library (200K entries)...")
    gen = LibraryGenerator(sim, chunk_size=50_000)
    lib_params, lib_signals = gen.generate(200_000, key=jax.random.PRNGKey(1))
    jax.block_until_ready(lib_signals)
    lib = SimulationLibrary(
        params=lib_params, signals=lib_signals,
        parameter_names=sim.parameter_names,
    )
    matcher = DictionaryMatcher(lib, k_best=10)
    print(f"Library: {lib.n_entries:,} entries, signal dim {lib.signal_dim}")

    # Build LM fitter
    fitter = build_lm_fitter(sim.forward_fn, acq)

    # JIT-compile the fitter for speed
    @jax.jit
    def lm_fit(data, init):
        return fitter.fit(data, acq, init)

    # Test crossing angles
    crossing_angles = np.arange(10, 95, 5)
    n_trials = 200
    snr = 30.0
    angle_threshold = 15.0

    results = {"angle": [], "dict_rate": [], "hybrid_rate": [], "lm_rate": []}

    key = jax.random.PRNGKey(42)

    for angle_deg in crossing_angles:
        angle_rad = np.radians(angle_deg)
        n_dict, n_hybrid, n_lm = 0, 0, 0

        for trial in range(n_trials):
            key, k_noise, k_init = jax.random.split(key, 3)

            mu1_true = np.array([0.0, 0.0, 1.0])
            mu2_true = np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
            f1 = 0.45

            signal = two_stick_signal(acq, jnp.array(mu1_true), jnp.array(mu2_true), f1)
            sigma = 1.0 / snr
            k1, k2 = jax.random.split(k_noise)
            n1 = jax.random.normal(k1, signal.shape) * sigma
            n2 = jax.random.normal(k2, signal.shape) * sigma
            noisy = jnp.sqrt((signal + n1) ** 2 + n2 ** 2)

            # --- Method 1: Dictionary matching ---
            dict_params, _ = matcher.match_single(noisy)
            dict_np = np.asarray(dict_params)
            mu1_d, mu2_d = params_to_orientations(dict_np)
            if check_both_detected(mu1_true, mu2_true, mu1_d, mu2_d, angle_threshold):
                n_dict += 1

            # --- Method 2: Dictionary + LM hybrid ---
            try:
                hybrid_params, _ = lm_fit(noisy, dict_params)
                hybrid_np = np.asarray(hybrid_params)
                mu1_h, mu2_h = params_to_orientations(hybrid_np)
                if check_both_detected(mu1_true, mu2_true, mu1_h, mu2_h, angle_threshold):
                    n_hybrid += 1
            except Exception:
                pass  # LM can diverge; count as failure

            # --- Method 3: Pure LM (random init) ---
            try:
                rand_init = sim.prior_sampler(k_init, 1)[0]
                lm_params, _ = lm_fit(noisy, rand_init)
                lm_np = np.asarray(lm_params)
                mu1_l, mu2_l = params_to_orientations(lm_np)
                if check_both_detected(mu1_true, mu2_true, mu1_l, mu2_l, angle_threshold):
                    n_lm += 1
            except Exception:
                pass

        d_rate = n_dict / n_trials
        h_rate = n_hybrid / n_trials
        l_rate = n_lm / n_trials
        results["angle"].append(angle_deg)
        results["dict_rate"].append(d_rate)
        results["hybrid_rate"].append(h_rate)
        results["lm_rate"].append(l_rate)
        print(f"  {angle_deg:3d} deg:  dict={d_rate:.0%}  hybrid={h_rate:.0%}  LM={l_rate:.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Peak Detection Rate by Crossing Angle")
    print("=" * 70)
    print(f"{'Angle':>6s}  {'Dictionary':>10s}  {'Dict+LM':>10s}  {'Pure LM':>10s}")
    for i, a in enumerate(results["angle"]):
        print(f"{a:>6d}  {results['dict_rate'][i]:>10.1%}  "
              f"{results['hybrid_rate'][i]:>10.1%}  {results['lm_rate'][i]:>10.1%}")

    np.savez(
        "validation/force_replication_results.npz",
        angles=results["angle"],
        dict_rates=results["dict_rate"],
        hybrid_rates=results["hybrid_rate"],
        lm_rates=results["lm_rate"],
    )
    print("\nResults saved to validation/force_replication_results.npz")


if __name__ == "__main__":
    main()
