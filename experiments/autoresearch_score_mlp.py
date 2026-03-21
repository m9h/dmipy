#!/usr/bin/env python3
"""
Autoresearch loop for MLP score-based diffusion posterior.

Round 1: Prove the denoising score matching objective works with plain MLPs.
Round 2: Scale up the best config.
Round 3: Compare against the flow baseline (3.2° at 200k steps).
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


@dataclass
class MLPScoreConfig:
    name: str = "baseline"
    hidden_dim: int = 256
    depth: int = 6
    cond_dim: int = 128
    learning_rate: float = 3e-4
    batch_size: int = 512
    n_steps: int = 5_000
    beta_min: float = 0.1
    beta_max: float = 20.0
    n_eval: int = 500
    n_posterior_samples: int = 200
    n_sde_steps: int = 100


def run_experiment(cfg: MLPScoreConfig, seed: int = 0):
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        MLPScoreNet, VPSchedule, train_score_posterior, sample_posterior,
    )

    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"  hidden={cfg.hidden_dim} depth={cfg.depth} lr={cfg.learning_rate} "
          f"bs={cfg.batch_size} steps={cfg.n_steps} "
          f"beta=({cfg.beta_min},{cfg.beta_max})")
    print(f"{'='*60}")

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
        theta = sim.prior_sampler(k, n)
        return (theta - lows) / spans

    score_net = MLPScoreNet(
        k_net,
        param_dim=sim.theta_dim,
        signal_dim=sim.signal_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        cond_dim=cfg.cond_dim,
    )
    schedule = VPSchedule(beta_min=cfg.beta_min, beta_max=cfg.beta_max)

    t0 = time.time()
    score_net, losses = train_score_posterior(
        k_train, score_net,
        simulator_fn=sim_fn, prior_fn=prior_fn, schedule=schedule,
        num_steps=cfg.n_steps, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        print_every=max(1, cfg.n_steps // 5),
    )
    train_time = time.time() - t0

    # Check if loss actually improved
    loss_delta = losses[-1] - losses[0]
    if loss_delta > -0.01:
        print(f"  WARNING: Loss did not improve ({losses[0]:.4f} → {losses[-1]:.4f})")

    # Evaluate
    sim_test = build_ball_2stick_simulator(acq, snr=30.0)
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, cfg.n_eval)

    @eqx.filter_jit
    def predict(net, sched, x):
        samples_norm = sample_posterior(
            jax.random.key(0), net, x, sched,
            n_samples=cfg.n_posterior_samples, n_steps=cfg.n_sde_steps,
            n_scalars=4, n_vectors=2,
        )
        samples = samples_norm * spans + lows
        samples = jnp.clip(samples, lows, highs)
        mu1 = samples[:, 4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1, axis=-1, keepdims=True), 1e-8)
        mu2 = samples[:, 7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2, axis=-1, keepdims=True), 1e-8)
        samples = samples.at[:, 4:7].set(mu1).at[:, 7:10].set(mu2)
        return jnp.median(samples, axis=0)

    preds = jax.vmap(predict, in_axes=(None, None, 0))(score_net, schedule, signals_test)
    preds_np, theta_np = np.asarray(preds), np.asarray(theta_test)

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
        "name": cfg.name,
        "final_loss": losses[-1],
        "loss_start": losses[0],
        "loss_delta": loss_delta,
        "fiber1_median_deg": float(np.median(errors1)),
        "fiber1_mean_deg": float(np.mean(errors1)),
        "d_stick_r": safe_r(theta_np[:, 1], preds_np[:, 1]),
        "f1_r": safe_r(theta_np[:, 2], preds_np[:, 2]),
        "d_ball_r": safe_r(theta_np[:, 0], preds_np[:, 0]),
        "train_time_s": train_time,
    }

    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={loss_delta:.4f})")
    print(f"  Fiber 1: {result['fiber1_median_deg']:.1f}° median")
    print(f"  d_stick r={result['d_stick_r']:.3f}  f1 r={result['f1_r']:.3f}")
    print(f"  Time: {train_time:.0f}s")

    return result


def main():
    results_dir = Path("experiments/autoresearch_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ---- Round 1: prove it works (5k steps, fast) ----
    print("\n" + "=" * 70)
    print("ROUND 1: MLP backbone search (5k steps each)")
    print("=" * 70)

    r1 = [
        MLPScoreConfig(name="mlp_small", hidden_dim=128, depth=4, n_steps=5000,
                       batch_size=256, n_eval=300, n_posterior_samples=100, n_sde_steps=50),
        MLPScoreConfig(name="mlp_medium", hidden_dim=256, depth=6, n_steps=5000,
                       batch_size=256, n_eval=300, n_posterior_samples=100, n_sde_steps=50),
        MLPScoreConfig(name="mlp_wide", hidden_dim=512, depth=4, n_steps=5000,
                       batch_size=256, n_eval=300, n_posterior_samples=100, n_sde_steps=50),
        MLPScoreConfig(name="mlp_deep", hidden_dim=256, depth=10, n_steps=5000,
                       batch_size=256, n_eval=300, n_posterior_samples=100, n_sde_steps=50),
        # LR sweep
        MLPScoreConfig(name="mlp_lr1e-3", hidden_dim=256, depth=6, learning_rate=1e-3,
                       n_steps=5000, batch_size=256, n_eval=300, n_posterior_samples=100,
                       n_sde_steps=50),
        MLPScoreConfig(name="mlp_lr1e-4", hidden_dim=256, depth=6, learning_rate=1e-4,
                       n_steps=5000, batch_size=256, n_eval=300, n_posterior_samples=100,
                       n_sde_steps=50),
        # Noise schedule sweep
        MLPScoreConfig(name="mlp_gentle", hidden_dim=256, depth=6, beta_min=0.01,
                       beta_max=10.0, n_steps=5000, batch_size=256, n_eval=300,
                       n_posterior_samples=100, n_sde_steps=50),
        MLPScoreConfig(name="mlp_aggressive", hidden_dim=256, depth=6, beta_min=0.1,
                       beta_max=40.0, n_steps=5000, batch_size=256, n_eval=300,
                       n_posterior_samples=100, n_sde_steps=50),
    ]

    for cfg in r1:
        try:
            all_results.append(run_experiment(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": cfg.name, "fiber1_median_deg": 90.0,
                                "error": str(e)})

    # Leaderboard
    print("\n" + "=" * 70)
    print("ROUND 1 LEADERBOARD")
    print("=" * 70)
    sorted_r1 = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    for i, r in enumerate(sorted_r1):
        print(f"  {i+1}. {r['name']:20s}  fiber1={r.get('fiber1_median_deg',99):.1f}°  "
              f"loss_Δ={r.get('loss_delta',0):.4f}  "
              f"d_stick_r={r.get('d_stick_r',0):.3f}  f1_r={r.get('f1_r',0):.3f}")

    with open(results_dir / "mlp_round1.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Round 2: scale up best ----
    best = sorted_r1[0]
    best_cfg = next(c for c in r1 if c.name == best["name"])
    print(f"\nBest: {best['name']} ({best.get('fiber1_median_deg',99):.1f}°)")

    print("\n" + "=" * 70)
    print("ROUND 2: Scale up best (10k-30k steps)")
    print("=" * 70)

    r2 = []
    for steps in [10_000, 30_000]:
        cfg = MLPScoreConfig(
            name=f"mlp_best_{steps//1000}k",
            hidden_dim=best_cfg.hidden_dim,
            depth=best_cfg.depth,
            cond_dim=best_cfg.cond_dim,
            learning_rate=best_cfg.learning_rate,
            beta_min=best_cfg.beta_min,
            beta_max=best_cfg.beta_max,
            batch_size=512,
            n_steps=steps,
            n_eval=500,
            n_posterior_samples=200,
            n_sde_steps=100,
        )
        r2.append(cfg)

    for cfg in r2:
        try:
            all_results.append(run_experiment(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": cfg.name, "fiber1_median_deg": 90.0,
                                "error": str(e)})

    # Final leaderboard
    all_sorted = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)
    for i, r in enumerate(all_sorted):
        print(f"  {i+1}. {r['name']:20s}  fiber1={r.get('fiber1_median_deg',99):.1f}°  "
              f"d_stick_r={r.get('d_stick_r',0):.3f}  "
              f"f1_r={r.get('f1_r',0):.3f}  "
              f"time={r.get('train_time_s',0):.0f}s")

    with open(results_dir / "mlp_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBest: {all_sorted[0]['name']} ({all_sorted[0].get('fiber1_median_deg',99):.1f}°)")
    print(f"Flow baseline: 3.2° at 200k steps")


if __name__ == "__main__":
    main()
