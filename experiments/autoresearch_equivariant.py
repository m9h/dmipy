#!/usr/bin/env python3
"""
Autoresearch: equivariant orientation head vs plain MLP.

Tests whether the local-frame decomposition (radial + tangential)
improves orientation estimation over the plain linear output.

Uses the best config from v3: v-prediction, DDPM, gentle schedule.
"""

import json, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, equinox as eqx

def run(name, net_cls, hidden_dim, n_steps, seed=0):
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        VPSchedule, train_score_posterior, sample_posterior,
    )

    print(f"\n{'='*60}")
    print(f"{name}  |  h={hidden_dim}  steps={n_steps}")

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

    if net_cls == "equivariant":
        from dmipy_jax.inference.score_posterior import MLPScoreNetEquivariant
        net = MLPScoreNetEquivariant(k_net, n_scalars=4, n_vectors=2,
                                      signal_dim=sim.signal_dim,
                                      hidden_dim=hidden_dim, depth=6)
    else:
        from dmipy_jax.inference.score_posterior import MLPScoreNet
        net = MLPScoreNet(k_net, param_dim=10, signal_dim=sim.signal_dim,
                          hidden_dim=hidden_dim, depth=6)

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
        "name": name,
        "fiber1_median_deg": float(np.median(errors1)),
        "d_stick_r": safe_r(theta_np[:, 1], preds_np[:, 1]),
        "f1_r": safe_r(theta_np[:, 2], preds_np[:, 2]),
        "final_loss": losses[-1],
        "train_time_s": train_time,
    }

    print(f"  → fiber1={result['fiber1_median_deg']:.1f}°  "
          f"d_stick_r={result['d_stick_r']:.3f}  f1_r={result['f1_r']:.3f}  "
          f"loss={losses[-1]:.4f}  time={train_time:.0f}s")
    return result


def main():
    results_dir = Path("experiments/autoresearch_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Head-to-head: equivariant vs plain MLP at same budget
    configs = [
        ("plain_mlp_10k",     "plain",       512, 10_000),
        ("equivariant_10k",   "equivariant", 512, 10_000),
        ("plain_mlp_30k",     "plain",       512, 30_000),
        ("equivariant_30k",   "equivariant", 512, 30_000),
    ]

    for name, cls, hdim, steps in configs:
        try:
            all_results.append(run(name, cls, hdim, steps))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "fiber1_median_deg": 90.0,
                                "error": str(e)})

    # Leaderboard
    all_sorted = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))
    print("\n" + "=" * 60)
    print("EQUIVARIANT HEAD LEADERBOARD")
    print("=" * 60)
    for i, r in enumerate(all_sorted):
        print(f"  {i+1}. {r['name']:25s}  fiber1={r.get('fiber1_median_deg',99):5.1f}°  "
              f"d_stick_r={r.get('d_stick_r',0):.3f}  "
              f"f1_r={r.get('f1_r',0):.3f}")

    with open(results_dir / "equivariant_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {results_dir}/equivariant_results.json")


if __name__ == "__main__":
    main()
