#!/usr/bin/env python3
"""
Autoresearch v3: Close the orientation gap.

Best so far: 15.5° with DDPM + gentle schedule + 512-wide MLP (30k steps).
Flow baseline: 3.2° at 200k steps.

Strategy:
  R1: v-prediction vs eps-prediction (v helps at low noise = fine features)
  R2: Network capacity (wider, deeper, bigger batch)
  R3: Long runs with best config (100k+ steps)
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
class Cfg:
    name: str = ""
    hidden_dim: int = 512
    depth: int = 6
    cond_dim: int = 128
    learning_rate: float = 3e-4
    batch_size: int = 512
    n_steps: int = 10_000
    beta_min: float = 0.01
    beta_max: float = 5.0
    prediction: str = "eps"   # "eps" or "v"
    sampler: str = "ddpm"
    n_sde_steps: int = 500
    n_eval: int = 500
    n_posterior_samples: int = 200


def run(cfg: Cfg, seed: int = 0):
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        MLPScoreNet, VPSchedule, train_score_posterior, sample_posterior,
    )

    print(f"\n{'='*60}")
    print(f"{cfg.name}  |  pred={cfg.prediction}  h={cfg.hidden_dim}  d={cfg.depth}  "
          f"lr={cfg.learning_rate}  steps={cfg.n_steps}  bs={cfg.batch_size}  "
          f"β=({cfg.beta_min},{cfg.beta_max})")

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

    net = MLPScoreNet(k_net, param_dim=sim.theta_dim, signal_dim=sim.signal_dim,
                      hidden_dim=cfg.hidden_dim, depth=cfg.depth, cond_dim=cfg.cond_dim)
    schedule = VPSchedule(beta_min=cfg.beta_min, beta_max=cfg.beta_max)

    t0 = time.time()
    net, losses = train_score_posterior(
        k_train, net, simulator_fn=sim_fn, prior_fn=prior_fn, schedule=schedule,
        num_steps=cfg.n_steps, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate, print_every=max(1, cfg.n_steps // 5),
        prediction=cfg.prediction,
    )
    train_time = time.time() - t0

    sim_test = build_ball_2stick_simulator(acq, snr=30.0)
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, cfg.n_eval)

    @eqx.filter_jit
    def predict(the_net, sched, x):
        samples_norm = sample_posterior(
            jax.random.key(0), the_net, x, sched,
            n_samples=cfg.n_posterior_samples, n_steps=cfg.n_sde_steps,
            n_scalars=4, n_vectors=2, method=cfg.sampler,
            prediction=cfg.prediction,
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
        "fiber1_median_deg": float(np.median(errors1)),
        "d_stick_r": safe_r(theta_np[:, 1], preds_np[:, 1]),
        "f1_r": safe_r(theta_np[:, 2], preds_np[:, 2]),
        "d_ball_r": safe_r(theta_np[:, 0], preds_np[:, 0]),
        "final_loss": losses[-1],
        "train_time_s": train_time,
        "config": asdict(cfg),
    }

    print(f"  → fiber1={result['fiber1_median_deg']:.1f}°  "
          f"d_stick_r={result['d_stick_r']:.3f}  f1_r={result['f1_r']:.3f}  "
          f"loss={losses[-1]:.4f}  time={train_time:.0f}s")
    return result


def main():
    results_dir = Path("experiments/autoresearch_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ================================================================
    # R1: v-prediction vs eps-prediction
    # ================================================================
    print("\n" + "=" * 70)
    print("R1: v-prediction vs eps-prediction (10k steps)")
    print("=" * 70)

    for pred in ["eps", "v"]:
        cfg = Cfg(name=f"r1_{pred}", prediction=pred, n_steps=10_000,
                  hidden_dim=512, depth=6, n_eval=300, n_posterior_samples=100)
        try:
            all_results.append(run(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": cfg.name, "fiber1_median_deg": 90.0, "error": str(e)})

    best_pred = min(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    best_prediction = best_pred.get("config", {}).get("prediction", "eps")
    print(f"\nBest prediction: {best_prediction} → {best_pred.get('fiber1_median_deg',99):.1f}°")

    # ================================================================
    # R2: Network capacity with best prediction type
    # ================================================================
    print("\n" + "=" * 70)
    print(f"R2: Network capacity sweep (pred={best_prediction}, 10k steps)")
    print("=" * 70)

    capacity_configs = [
        Cfg(name="r2_w256_d6", hidden_dim=256, depth=6, prediction=best_prediction,
            n_steps=10_000, n_eval=300, n_posterior_samples=100),
        Cfg(name="r2_w512_d6", hidden_dim=512, depth=6, prediction=best_prediction,
            n_steps=10_000, n_eval=300, n_posterior_samples=100),
        Cfg(name="r2_w512_d10", hidden_dim=512, depth=10, prediction=best_prediction,
            n_steps=10_000, n_eval=300, n_posterior_samples=100),
        Cfg(name="r2_w1024_d6", hidden_dim=1024, depth=6, prediction=best_prediction,
            n_steps=10_000, n_eval=300, n_posterior_samples=100),
        # Larger batch
        Cfg(name="r2_w512_bs1024", hidden_dim=512, depth=6, batch_size=1024,
            prediction=best_prediction, n_steps=10_000, n_eval=300, n_posterior_samples=100),
    ]
    for cfg in capacity_configs:
        try:
            all_results.append(run(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": cfg.name, "fiber1_median_deg": 90.0, "error": str(e)})

    # ================================================================
    # R3: Long run with best config
    # ================================================================
    best_all = min(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    bc = best_all.get("config", {})
    print(f"\nBest so far: {best_all['name']} → {best_all.get('fiber1_median_deg',99):.1f}°")

    print("\n" + "=" * 70)
    print("R3: Long runs (50k-100k steps)")
    print("=" * 70)

    for n_steps in [50_000, 100_000]:
        cfg = Cfg(
            name=f"r3_{n_steps//1000}k",
            hidden_dim=bc.get("hidden_dim", 512),
            depth=bc.get("depth", 6),
            cond_dim=bc.get("cond_dim", 128),
            learning_rate=bc.get("learning_rate", 3e-4),
            batch_size=bc.get("batch_size", 512),
            prediction=bc.get("prediction", best_prediction),
            beta_min=bc.get("beta_min", 0.01),
            beta_max=bc.get("beta_max", 5.0),
            n_steps=n_steps,
            n_sde_steps=500,
            n_eval=500,
            n_posterior_samples=200,
        )
        try:
            all_results.append(run(cfg))
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"name": cfg.name, "fiber1_median_deg": 90.0, "error": str(e)})

    # ================================================================
    # Final
    # ================================================================
    all_sorted = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)
    for i, r in enumerate(all_sorted):
        print(f"  {i+1:2d}. {r['name']:22s}  fiber1={r.get('fiber1_median_deg',99):5.1f}°  "
              f"d_stick_r={r.get('d_stick_r',0):.3f}  "
              f"f1_r={r.get('f1_r',0):.3f}  "
              f"time={r.get('train_time_s',0):.0f}s")
    print(f"\nFlow baseline: 3.2° at 200k steps")

    with open(results_dir / "v3_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
