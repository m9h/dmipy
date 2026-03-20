#!/usr/bin/env python3
"""
Autoresearch loop for score-based diffusion posterior estimation.

Inspired by Karpathy's autoresearch: define a search space of architectural
choices, run short experiments, evaluate, and iterate.

Each experiment trains for a short budget (5k steps, ~3 min), evaluates
orientation error, and logs to trackio. After all experiments in a round,
the best config is identified and used as the basis for the next round.

Usage:
    python experiments/autoresearch_score.py
"""

import json
import time
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ScoreExperimentConfig:
    """One experimental configuration to evaluate."""
    name: str = "baseline"

    # Architecture
    hidden_scalars: int = 64
    hidden_vectors: int = 21
    depth: int = 4
    cond_dim: int = 128
    use_residual: bool = False
    cond_injection: str = "film"  # "film", "concat", "tensor_product"

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 512
    n_steps: int = 5_000
    beta_min: float = 0.1
    beta_max: float = 20.0

    # Eval
    n_eval: int = 500
    n_posterior_samples: int = 200
    n_sde_steps: int = 100


# ---------------------------------------------------------------------------
# Modified score network with architectural variants
# ---------------------------------------------------------------------------

def build_score_net(cfg: ScoreExperimentConfig, key, signal_dim: int,
                    n_scalars: int = 4, n_vectors: int = 2):
    """Build a score network according to the experiment config."""
    import e3nn_jax as e3nn

    from dmipy_jax.inference.score_posterior import (
        E3Linear, FiLMModulation, VPSchedule,
    )

    param_irreps = f"{n_scalars}x0e + {n_vectors}x1o"
    hidden_irreps = f"{cfg.hidden_scalars}x0e + {cfg.hidden_vectors}x1o"

    # Extra 0e channels for concat conditioning
    if cfg.cond_injection == "concat":
        # Add cond_dim extra scalars to hidden representation
        input_with_cond = f"{n_scalars + cfg.cond_dim}x0e + {n_vectors}x1o"
        hidden_with_cond = f"{cfg.hidden_scalars + cfg.cond_dim}x0e + {cfg.hidden_vectors}x1o"
    else:
        input_with_cond = param_irreps
        hidden_with_cond = hidden_irreps

    keys = jax.random.split(key, 2 * cfg.depth + 5)

    # Conditioning encoders
    signal_encoder = eqx.nn.MLP(
        in_size=signal_dim, out_size=cfg.cond_dim, width_size=cfg.cond_dim,
        depth=2, activation=jax.nn.gelu, key=keys[0],
    )
    time_encoder = eqx.nn.MLP(
        in_size=1, out_size=cfg.cond_dim, width_size=cfg.cond_dim,
        depth=2, activation=jax.nn.silu, key=keys[1],
    )

    layers = []
    film_layers = []

    # Input layer
    in_irreps = input_with_cond if cfg.cond_injection == "concat" else param_irreps
    layers.append(E3Linear(in_irreps, hidden_irreps, keys[2]))
    if cfg.cond_injection == "film":
        film_layers.append(FiLMModulation(cfg.cond_dim * 2, cfg.hidden_scalars, keys[3]))

    # Hidden layers
    for i in range(cfg.depth - 1):
        h_in = hidden_with_cond if cfg.cond_injection == "concat" else hidden_irreps
        layers.append(E3Linear(h_in, hidden_irreps, keys[4 + 2 * i]))
        if cfg.cond_injection == "film":
            film_layers.append(FiLMModulation(cfg.cond_dim * 2, cfg.hidden_scalars, keys[5 + 2 * i]))

    # Output
    layers.append(E3Linear(hidden_irreps, param_irreps, keys[-1]))

    return _FlexScoreNet(
        layers=layers,
        film_layers=film_layers,
        signal_encoder=signal_encoder,
        time_encoder=time_encoder,
        param_irreps=param_irreps,
        hidden_irreps=hidden_irreps,
        n_hidden_scalars=cfg.hidden_scalars,
        cond_injection=cfg.cond_injection,
        cond_dim=cfg.cond_dim,
        use_residual=cfg.use_residual,
    )


class _FlexScoreNet(eqx.Module):
    """Flexible score net supporting multiple conditioning strategies."""
    layers: list
    film_layers: list
    signal_encoder: eqx.nn.MLP
    time_encoder: eqx.nn.MLP
    param_irreps: str
    hidden_irreps: str
    n_hidden_scalars: int
    cond_injection: str
    cond_dim: int
    use_residual: bool

    def __call__(self, theta_t, t, signal):
        import e3nn_jax as e3nn

        sig_emb = self.signal_encoder(signal)
        t_emb = self.time_encoder(jnp.atleast_1d(t))
        cond = jnp.concatenate([sig_emb, t_emb])

        if self.cond_injection == "concat":
            # Prepend conditioning as extra 0e channels
            x_arr = jnp.concatenate([theta_t, cond])
            in_irreps = f"{self.param_irreps.split('+')[0].strip().split('x')[0]}x0e"
            # Re-count: n_scalars from param + cond_dim
            n_param_scalars = int(self.param_irreps.split("x0e")[0].strip())
            full_in = f"{n_param_scalars + self.cond_dim}x0e"
            if "1o" in self.param_irreps:
                vec_part = self.param_irreps.split("+")[1].strip()
                full_in = f"{full_in} + {vec_part}"
            x = e3nn.IrrepsArray(full_in, x_arr)
        else:
            x = e3nn.IrrepsArray(self.param_irreps, theta_t)

        for i, layer in enumerate(self.layers[:-1]):
            x_in = x
            x = layer(x)

            if self.cond_injection == "film" and i < len(self.film_layers):
                x = self._film_and_activate(x, cond, self.film_layers[i])
            else:
                n_groups = len(e3nn.Irreps(x.irreps))
                acts = [jax.nn.gelu] * n_groups
                x = e3nn.norm_activation(x, acts)

            # Residual connection (if shapes match)
            if self.use_residual and i > 0 and x.array.shape == x_in.array.shape:
                x = e3nn.IrrepsArray(x.irreps, x.array + x_in.array)

            # Re-concat conditioning for next layer if using concat mode
            if self.cond_injection == "concat" and i < len(self.layers) - 2:
                arr = jnp.concatenate([x.array[:self.n_hidden_scalars], cond,
                                       x.array[self.n_hidden_scalars:]])
                full_h = f"{self.n_hidden_scalars + self.cond_dim}x0e"
                if "1o" in self.hidden_irreps:
                    vec_part = self.hidden_irreps.split("+")[1].strip()
                    full_h = f"{full_h} + {vec_part}"
                x = e3nn.IrrepsArray(full_h, arr)

        x = self.layers[-1](x)
        return x.array

    def _film_and_activate(self, x, cond, film):
        import e3nn_jax as e3nn
        gamma, beta = film(cond)
        arr = x.array
        ns = self.n_hidden_scalars
        scalars = arr[:ns] * (1.0 + gamma) + beta
        vectors = arr[ns:]
        arr = jnp.concatenate([scalars, vectors])
        x = e3nn.IrrepsArray(x.irreps, arr)
        n_groups = len(e3nn.Irreps(x.irreps))
        acts = [jax.nn.gelu] * n_groups
        x = e3nn.norm_activation(x, acts)
        return x


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: ScoreExperimentConfig, seed: int = 0):
    """Run one experiment and return metrics."""
    from validation.validate_nottingham_sbi import (
        make_hcp_like_acquisition, build_ball_2stick_simulator,
        angular_error_deg_cartesian,
    )
    from dmipy_jax.inference.score_posterior import (
        VPSchedule, train_score_posterior, sample_posterior,
    )

    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"{'='*60}")
    print(json.dumps({k: v for k, v in asdict(cfg).items()
                      if k != "name"}, indent=2))

    key = jax.random.key(seed)
    k_net, k_train, k_eval = jax.random.split(key, 3)

    acq = make_hcp_like_acquisition()
    sim = build_ball_2stick_simulator(acq, snr=30.0)
    sim.snr_range = (10.0, 50.0)

    # Normalisation bounds
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

    # Build network
    score_net = build_score_net(cfg, k_net, signal_dim=sim.signal_dim)
    schedule = VPSchedule(beta_min=cfg.beta_min, beta_max=cfg.beta_max)

    # Train
    t0 = time.time()
    score_net, losses = train_score_posterior(
        k_train, score_net,
        simulator_fn=sim_fn, prior_fn=prior_fn, schedule=schedule,
        num_steps=cfg.n_steps, batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate, print_every=cfg.n_steps // 5,
    )
    train_time = time.time() - t0

    # Evaluate
    sim_test = build_ball_2stick_simulator(acq, snr=30.0)
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, cfg.n_eval)

    @eqx.filter_jit
    def predict(net, sched, x):
        samples_norm = sample_posterior(
            jax.random.key(0), net, x, sched,
            n_samples=cfg.n_posterior_samples, n_steps=cfg.n_sde_steps,
        )
        samples = samples_norm * spans + lows
        samples = jnp.clip(samples, lows, highs)
        mu1 = samples[:, 4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1, axis=-1, keepdims=True), 1e-8)
        mu2 = samples[:, 7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2, axis=-1, keepdims=True), 1e-8)
        samples = samples.at[:, 4:7].set(mu1).at[:, 7:10].set(mu2)
        return jnp.median(samples, axis=0)

    # Normalise test signals
    preds = jax.vmap(predict, in_axes=(None, None, 0))(
        score_net, schedule, signals_test
    )
    preds_np, theta_np = np.asarray(preds), np.asarray(theta_test)

    # Metrics
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

    d_stick_r = np.corrcoef(theta_np[:, 1], preds_np[:, 1])[0, 1]
    f1_r = np.corrcoef(theta_np[:, 2], preds_np[:, 2])[0, 1]

    result = {
        "name": cfg.name,
        "final_loss": losses[-1],
        "fiber1_median_deg": float(np.median(errors1)),
        "fiber1_mean_deg": float(np.mean(errors1)),
        "d_stick_r": float(d_stick_r) if not np.isnan(d_stick_r) else 0.0,
        "f1_r": float(f1_r) if not np.isnan(f1_r) else 0.0,
        "train_time_s": train_time,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_improved": losses[-1] < losses[0] - 0.01,
    }

    print(f"\n--- Results: {cfg.name} ---")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Fiber 1 median: {result['fiber1_median_deg']:.1f} deg")
    print(f"  d_stick r: {result['d_stick_r']:.4f}")
    print(f"  f1 r: {result['f1_r']:.4f}")
    print(f"  Time: {train_time:.0f}s")

    return result


# ---------------------------------------------------------------------------
# Autoresearch loop
# ---------------------------------------------------------------------------

def round_1_experiments():
    """Round 1: isolate which conditioning/architecture changes help."""
    base = dict(n_steps=5_000, batch_size=256, n_eval=300,
                n_posterior_samples=100, n_sde_steps=50)
    return [
        # Baseline (original failing config)
        ScoreExperimentConfig(name="r1_baseline", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="film",
                              use_residual=False, **base),
        # Fix 1: concat conditioning (signal info reaches all channels)
        ScoreExperimentConfig(name="r1_concat", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="concat",
                              use_residual=False, **base),
        # Fix 2: residual connections
        ScoreExperimentConfig(name="r1_residual", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="film",
                              use_residual=True, **base),
        # Fix 3: concat + residual
        ScoreExperimentConfig(name="r1_concat_res", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="concat",
                              use_residual=True, **base),
        # Fix 4: bigger network
        ScoreExperimentConfig(name="r1_big", hidden_scalars=128,
                              hidden_vectors=42, depth=4, cond_injection="film",
                              use_residual=False, **base),
        # Fix 5: bigger + concat + residual
        ScoreExperimentConfig(name="r1_big_concat_res", hidden_scalars=128,
                              hidden_vectors=42, depth=4, cond_injection="concat",
                              use_residual=True, **base),
        # Fix 6: higher learning rate
        ScoreExperimentConfig(name="r1_high_lr", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="concat",
                              use_residual=True, learning_rate=1e-3, **base),
        # Fix 7: gentler noise schedule
        ScoreExperimentConfig(name="r1_gentle_sched", hidden_scalars=64,
                              hidden_vectors=21, depth=4, cond_injection="concat",
                              use_residual=True, beta_min=0.01, beta_max=10.0,
                              **base),
    ]


def round_2_experiments(best_cfg: dict):
    """Round 2: refine the best config from round 1 with longer training."""
    configs = []
    # Take the best and try longer training + variations
    for steps in [10_000, 20_000]:
        for depth in [4, 6]:
            cfg = ScoreExperimentConfig(
                name=f"r2_steps{steps//1000}k_d{depth}",
                n_steps=steps, depth=depth, batch_size=512,
                n_eval=500, n_posterior_samples=200, n_sde_steps=100,
                **{k: v for k, v in best_cfg.items()
                   if k in ("hidden_scalars", "hidden_vectors", "cond_injection",
                            "use_residual", "learning_rate", "beta_min", "beta_max",
                            "cond_dim")},
            )
            configs.append(cfg)
    return configs


def main():
    results_dir = Path("experiments/autoresearch_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Round 1 ----
    print("\n" + "=" * 70)
    print("AUTORESEARCH ROUND 1: Architecture search (5k steps each)")
    print("=" * 70)

    r1_configs = round_1_experiments()
    r1_results = []

    for cfg in r1_configs:
        try:
            result = run_experiment(cfg)
            r1_results.append(result)
        except Exception as e:
            print(f"FAILED: {cfg.name}: {e}")
            r1_results.append({"name": cfg.name, "fiber1_median_deg": 90.0,
                               "loss_improved": False, "error": str(e)})

    # Leaderboard
    print("\n" + "=" * 70)
    print("ROUND 1 LEADERBOARD")
    print("=" * 70)
    r1_sorted = sorted(r1_results, key=lambda x: x.get("fiber1_median_deg", 90))
    for i, r in enumerate(r1_sorted):
        marker = " ★" if i == 0 else ""
        improved = "✓" if r.get("loss_improved") else "✗"
        print(f"  {i+1}. {r['name']:25s}  "
              f"fiber1={r.get('fiber1_median_deg', 99):.1f}°  "
              f"loss_improved={improved}  "
              f"d_stick_r={r.get('d_stick_r', 0):.3f}{marker}")

    # Save round 1
    with open(results_dir / "round1.json", "w") as f:
        json.dump(r1_results, f, indent=2)

    # ---- Round 2: refine best ----
    best_r1 = r1_sorted[0]
    best_r1_cfg = next(c for c in r1_configs if c.name == best_r1["name"])
    best_dict = asdict(best_r1_cfg)

    print(f"\nBest from round 1: {best_r1['name']}")
    print(f"  fiber1={best_r1.get('fiber1_median_deg', 99):.1f}°")

    print("\n" + "=" * 70)
    print("AUTORESEARCH ROUND 2: Scale up best config (10-20k steps)")
    print("=" * 70)

    r2_configs = round_2_experiments(best_dict)
    r2_results = []

    for cfg in r2_configs:
        try:
            result = run_experiment(cfg)
            r2_results.append(result)
        except Exception as e:
            print(f"FAILED: {cfg.name}: {e}")
            r2_results.append({"name": cfg.name, "fiber1_median_deg": 90.0,
                               "error": str(e)})

    # Final leaderboard
    all_results = r1_results + r2_results
    all_sorted = sorted(all_results, key=lambda x: x.get("fiber1_median_deg", 90))

    print("\n" + "=" * 70)
    print("FINAL LEADERBOARD (ALL ROUNDS)")
    print("=" * 70)
    for i, r in enumerate(all_sorted[:10]):
        print(f"  {i+1}. {r['name']:25s}  "
              f"fiber1={r.get('fiber1_median_deg', 99):.1f}°  "
              f"d_stick_r={r.get('d_stick_r', 0):.3f}  "
              f"f1_r={r.get('f1_r', 0):.3f}  "
              f"time={r.get('train_time_s', 0):.0f}s")

    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print(f"Best config: {all_sorted[0]['name']} "
          f"({all_sorted[0].get('fiber1_median_deg', 99):.1f}°)")


if __name__ == "__main__":
    main()
