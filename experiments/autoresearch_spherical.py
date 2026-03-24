#!/usr/bin/env python3
"""
Autoresearch: spherical vs Cartesian orientation parameterization.

The Nottingham paper (Manzano-Patron et al.) showed that spherical coordinates
(theta, phi) with Jacobian-corrected prior achieves ~2 deg orientation error,
while Cartesian (x,y,z) wastes network capacity learning the unit-sphere
constraint.

This experiment compares:
  - **Cartesian (baseline)**: 10-D parameter vector [d_ball, d_stick, f1, f2,
    mu1x, mu1y, mu1z, mu2x, mu2y, mu2z] with MLPScoreNet
  - **Spherical**: 8-D parameter vector [d_ball, d_stick, f1, f2, theta1, phi1,
    theta2, phi2] with MLPScoreNetSpherical, Jacobian-corrected angular prior,
    and sim_fn that converts spherical -> Cartesian before the forward model

Uses best config from v3: v-prediction, DDPM, gentle schedule
(beta_min=0.01, beta_max=5.0), 512-wide, depth 6.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def run_cartesian(name, n_steps, seed=0):
    """Run the Cartesian (10-D) baseline."""
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        MLPScoreNet, VPSchedule, train_score_posterior, sample_posterior,
    )

    print(f"\n{'='*60}")
    print(f"{name}  |  Cartesian 10-D  |  steps={n_steps}")
    print("=" * 60)

    key = jax.random.key(seed)
    k_net, k_train, k_eval = jax.random.split(key, 3)

    acq = make_hcp_like_acquisition()
    sim = build_ball_2stick_simulator(acq, snr=30.0)
    sim.snr_range = (10.0, 50.0)

    lows = jnp.array([sim.parameter_ranges[n][0] for n in sim.parameter_names])
    highs = jnp.array([sim.parameter_ranges[n][1] for n in sim.parameter_names])
    spans = jnp.maximum(highs - lows, 1e-12)
    b0_mask = sim.acquisition.bvalues < 100e6

    def sim_fn(k, theta_norm):
        k1, k2 = jax.random.split(k)
        theta = theta_norm * spans + lows
        signal = sim.simulate(k1, theta)
        noisy = sim.add_noise(k2, signal)
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean
        return noisy

    def prior_fn(k, n):
        return (sim.prior_sampler(k, n) - lows) / spans

    net = MLPScoreNet(k_net, param_dim=10, signal_dim=sim.signal_dim,
                      hidden_dim=512, depth=6, cond_dim=128)
    schedule = VPSchedule(beta_min=0.01, beta_max=5.0)

    t0 = time.time()
    net, losses = train_score_posterior(
        k_train, net, simulator_fn=sim_fn, prior_fn=prior_fn, schedule=schedule,
        num_steps=n_steps, batch_size=512, learning_rate=3e-4,
        print_every=max(1, n_steps // 5), prediction="v",
    )
    train_time = time.time() - t0

    # Evaluate
    sim_test = build_ball_2stick_simulator(acq, snr=30.0)
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, 500)

    @eqx.filter_jit
    def predict(the_net, sched, x):
        samples_norm = sample_posterior(
            jax.random.key(0), the_net, x, sched,
            n_samples=200, n_steps=500, n_scalars=4, n_vectors=2,
            method="ddpm", prediction="v",
        )
        samples = samples_norm * spans + lows
        samples = jnp.clip(samples, lows, highs)
        mu1 = samples[:, 4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1, axis=-1, keepdims=True), 1e-8)
        mu2 = samples[:, 7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2, axis=-1, keepdims=True), 1e-8)
        samples = samples.at[:, 4:7].set(mu1).at[:, 7:10].set(mu2)
        return jnp.median(samples, axis=0)

    preds = jax.vmap(predict, in_axes=(None, None, 0))(net, schedule, signals_test)
    return _evaluate(name, preds, theta_test, losses, train_time, angular_error_deg_cartesian)


def run_spherical(name, n_steps, seed=0):
    """Run the spherical (8-D) parameterization.

    Key differences from Cartesian:
      - Prior uses Jacobian-corrected angular sampling:
          theta = arccos(1 - 2u), u ~ U(0,1)  ->  uniform on sphere
          phi = pi * u                          ->  U(0, pi) for antipodal
      - sim_fn converts spherical params back to Cartesian for the forward model
      - After DDPM sampling, angles are clamped and converted to Cartesian
    """
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        MLPScoreNetSpherical, VPSchedule, train_score_posterior,
        sample_posterior, spherical_to_cartesian_batch,
        cartesian_to_spherical_batch,
    )

    print(f"\n{'='*60}")
    print(f"{name}  |  Spherical 8-D  |  steps={n_steps}")
    print("=" * 60)

    key = jax.random.key(seed)
    k_net, k_train, k_eval = jax.random.split(key, 3)

    acq = make_hcp_like_acquisition()
    sim = build_ball_2stick_simulator(acq, snr=30.0)
    sim.snr_range = (10.0, 50.0)

    # Cartesian parameter ranges (for the 10-D model)
    lows_cart = jnp.array([sim.parameter_ranges[n][0] for n in sim.parameter_names])
    highs_cart = jnp.array([sim.parameter_ranges[n][1] for n in sim.parameter_names])
    spans_cart = jnp.maximum(highs_cart - lows_cart, 1e-12)

    # Spherical parameter ranges (8-D)
    # [d_ball, d_stick, f1, f2, theta1, phi1, theta2, phi2]
    # Scalar ranges are same as Cartesian; angles have natural ranges
    scalar_lows = lows_cart[:4]
    scalar_highs = highs_cart[:4]
    scalar_spans = spans_cart[:4]

    # Angle ranges for normalization to [0, 1]
    # theta in [0, pi/2], phi in [0, pi)
    angle_lows = jnp.array([0.0, 0.0, 0.0, 0.0])   # theta1, phi1, theta2, phi2
    angle_highs = jnp.array([jnp.pi / 2, jnp.pi, jnp.pi / 2, jnp.pi])
    angle_spans = angle_highs - angle_lows

    # Full 8-D normalization arrays
    lows_sph = jnp.concatenate([scalar_lows, angle_lows])
    highs_sph = jnp.concatenate([scalar_highs, angle_highs])
    spans_sph = jnp.concatenate([scalar_spans, angle_spans])

    b0_mask = sim.acquisition.bvalues < 100e6

    def spherical_prior_fn(k, n):
        """Jacobian-corrected prior in spherical coordinates.

        For uniform distribution on the hemisphere:
          theta = arccos(1 - u) where u ~ U(0,1)  (gives uniform in z >= 0)
          phi = pi * v where v ~ U(0,1)            (antipodal: [0, pi))

        The scalar parameters use the same uniform prior as Cartesian.
        """
        keys = jax.random.split(k, 7)

        # Scalar prior (same as Cartesian)
        d_ball = jax.random.uniform(keys[0], (n,), minval=1.0e-9, maxval=3.5e-9)
        d_stick = jax.random.uniform(keys[1], (n,), minval=0.5e-9, maxval=2.5e-9)
        f1_raw = jax.random.uniform(keys[2], (n,), minval=0.1, maxval=0.8)
        f2_raw = jax.random.uniform(keys[3], (n,), minval=0.05, maxval=0.5)
        total = f1_raw + f2_raw
        scale = jnp.minimum(1.0, 0.95 / total)
        f1 = f1_raw * scale
        f2 = f2_raw * scale

        # Canonical ordering: f1 >= f2
        swap = f1 < f2
        f1_out = jnp.where(swap, f2, f1)
        f2_out = jnp.where(swap, f1, f2)

        # Jacobian-corrected angular prior:
        # theta = arccos(1 - u), u ~ U(0,1) gives density proportional to sin(theta)
        # This ensures uniform sampling on the hemisphere surface.
        # Since we want z >= 0 (theta in [0, pi/2]), use u ~ U(0,1):
        #   theta = arccos(1 - u)  ranges from 0 (u=0) to pi/2 (u=1)
        u1 = jax.random.uniform(keys[4], (n,))
        theta1 = jnp.arccos(1.0 - u1)  # [0, pi/2]
        phi1 = jax.random.uniform(keys[5], (n,), minval=0.0, maxval=jnp.pi)

        u2 = jax.random.uniform(keys[4], (n,))  # reuse key branch for fiber 2
        k_ang2a, k_ang2b = jax.random.split(keys[6])
        u2 = jax.random.uniform(k_ang2a, (n,))
        theta2 = jnp.arccos(1.0 - u2)  # [0, pi/2]
        phi2 = jax.random.uniform(k_ang2b, (n,), minval=0.0, maxval=jnp.pi)

        # Swap angles with fractions to maintain correspondence
        theta1_out = jnp.where(swap, theta2, theta1)
        phi1_out = jnp.where(swap, phi2, phi1)
        theta2_out = jnp.where(swap, theta1, theta2)
        phi2_out = jnp.where(swap, phi1, phi2)

        # 8-D parameter vector
        theta_sph = jnp.stack([
            d_ball, d_stick, f1_out, f2_out,
            theta1_out, phi1_out, theta2_out, phi2_out,
        ], axis=-1)

        # Normalize to [0, 1]
        return (theta_sph - lows_sph) / spans_sph

    def sim_fn_spherical(k, theta_norm):
        """Convert 8-D spherical (normalized) to 10-D Cartesian, then simulate."""
        k1, k2 = jax.random.split(k)

        # Denormalize 8-D
        theta_sph = theta_norm * spans_sph + lows_sph

        # Extract scalars and angles
        scalars = theta_sph[:, :4]  # (batch, 4)
        theta1 = theta_sph[:, 4]
        phi1 = theta_sph[:, 5]
        theta2 = theta_sph[:, 6]
        phi2 = theta_sph[:, 7]

        # Convert angles to Cartesian unit vectors
        mu1 = spherical_to_cartesian_batch(theta1, phi1)  # (batch, 3)
        mu2 = spherical_to_cartesian_batch(theta2, phi2)  # (batch, 3)

        # Assemble 10-D Cartesian parameter vector
        theta_cart = jnp.concatenate([scalars, mu1, mu2], axis=-1)  # (batch, 10)

        # Forward model expects raw (un-normalized) Cartesian params
        signal = sim.simulate(k1, theta_cart)
        noisy = sim.add_noise(k2, signal)
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean
        return noisy

    net = MLPScoreNetSpherical(k_net, param_dim=8, signal_dim=sim.signal_dim,
                                hidden_dim=512, depth=6, cond_dim=128)
    schedule = VPSchedule(beta_min=0.01, beta_max=5.0)

    t0 = time.time()
    net, losses = train_score_posterior(
        k_train, net, simulator_fn=sim_fn_spherical,
        prior_fn=spherical_prior_fn, schedule=schedule,
        num_steps=n_steps, batch_size=512, learning_rate=3e-4,
        print_every=max(1, n_steps // 5), prediction="v",
    )
    train_time = time.time() - t0

    # Evaluate — ground truth is still 10-D Cartesian
    sim_test = build_ball_2stick_simulator(acq, snr=30.0)
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, 500)

    @eqx.filter_jit
    def predict_spherical(the_net, sched, x):
        # sample_posterior with spherical=True returns 10-D:
        # [normalized_scalars(4), unit_mu1(3), unit_mu2(3)]
        # Scalars are still normalized [0,1]; angles were already converted
        # to Cartesian unit vectors inside _sample_ddpm_spherical.
        samples_cart = sample_posterior(
            jax.random.key(0), the_net, x, sched,
            n_samples=200, n_steps=500, n_scalars=4, n_vectors=2,
            method="ddpm", prediction="v", spherical=True,
        )

        # Denormalize scalars
        raw_scalars = samples_cart[:, :4] * scalar_spans + scalar_lows
        raw_scalars = jnp.clip(raw_scalars, scalar_lows, scalar_highs)

        # Re-normalize orientation vectors
        mu1 = samples_cart[:, 4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1, axis=-1, keepdims=True), 1e-8)
        mu2 = samples_cart[:, 7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2, axis=-1, keepdims=True), 1e-8)

        samples_out = jnp.concatenate([raw_scalars, mu1, mu2], axis=-1)
        return jnp.median(samples_out, axis=0)

    preds = jax.vmap(predict_spherical, in_axes=(None, None, 0))(net, schedule, signals_test)
    return _evaluate(name, preds, theta_test, losses, train_time, angular_error_deg_cartesian)


def _evaluate(name, preds, theta_test, losses, train_time, angular_error_deg_cartesian):
    """Compute metrics (shared between Cartesian and spherical runs)."""
    preds_np = np.asarray(preds)
    theta_np = np.asarray(theta_test)

    errors1 = []
    for j in range(len(theta_np)):
        m1t = theta_np[j, 4:7]
        m1p = preds_np[j, 4:7] / max(np.linalg.norm(preds_np[j, 4:7]), 1e-8)
        m2t = theta_np[j, 7:10]
        m2p = preds_np[j, 7:10] / max(np.linalg.norm(preds_np[j, 7:10]), 1e-8)
        e11 = angular_error_deg_cartesian(m1t, m1p)
        e22 = angular_error_deg_cartesian(m2t, m2p)
        e12 = angular_error_deg_cartesian(m1t, m2p)
        e21 = angular_error_deg_cartesian(m2t, m1p)
        errors1.append(min(e11, e12))
    errors1 = np.array(errors1)

    def safe_r(a, b):
        r = np.corrcoef(a, b)[0, 1]
        return float(r) if not np.isnan(r) else 0.0

    result = {
        "name": name,
        "fiber1_median_deg": float(np.median(errors1)),
        "fiber1_mean_deg": float(np.mean(errors1)),
        "d_stick_r": safe_r(theta_np[:, 1], preds_np[:, 1]),
        "f1_r": safe_r(theta_np[:, 2], preds_np[:, 2]),
        "d_ball_r": safe_r(theta_np[:, 0], preds_np[:, 0]),
        "final_loss": float(losses[-1]),
        "train_time_s": train_time,
    }

    print(f"  -> fiber1={result['fiber1_median_deg']:.1f} deg (median)  "
          f"mean={result['fiber1_mean_deg']:.1f} deg  "
          f"d_stick_r={result['d_stick_r']:.3f}  f1_r={result['f1_r']:.3f}  "
          f"loss={result['final_loss']:.4f}  time={train_time:.0f}s")
    return result


def main():
    results_dir = Path("experiments/autoresearch_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ================================================================
    # Head-to-head: Cartesian vs Spherical at 10k and 30k steps
    # ================================================================
    configs = [
        ("cartesian_10k",  "cartesian", 10_000),
        ("spherical_10k",  "spherical", 10_000),
        ("cartesian_30k",  "cartesian", 30_000),
        ("spherical_30k",  "spherical", 30_000),
    ]

    for name, param_type, n_steps in configs:
        try:
            if param_type == "cartesian":
                result = run_cartesian(name, n_steps)
            else:
                result = run_spherical(name, n_steps)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "name": name,
                "fiber1_median_deg": 90.0,
                "error": str(e),
            })

    # ================================================================
    # Leaderboard
    # ================================================================
    all_sorted = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    print("\n" + "=" * 70)
    print("SPHERICAL vs CARTESIAN LEADERBOARD")
    print("=" * 70)
    for i, r in enumerate(all_sorted):
        print(f"  {i+1}. {r['name']:25s}  fiber1={r.get('fiber1_median_deg',99):5.1f} deg  "
              f"d_stick_r={r.get('d_stick_r',0):.3f}  "
              f"f1_r={r.get('f1_r',0):.3f}  "
              f"time={r.get('train_time_s',0):.0f}s")

    print(f"\nFlow baseline: 3.2 deg at 200k steps")

    out_path = results_dir / "spherical_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
