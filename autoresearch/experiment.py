#!/usr/bin/env python3
"""Baseline experiment: MLP score-based posterior for Ball + 2-Stick.

This is the root experiment for agentsciml evolutionary search.
It trains a plain MLP denoising score network and evaluates fiber
orientation error on synthetic HCP-like data.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from prepare import (
    ExperimentResult,
    get_commit_hash,
    print_result,
    log_result,
    make_hcp_acquisition,
    build_simulator,
    compute_fiber_errors,
    safe_pearson_r,
)


def run_experiment(
    hidden_dim: int = 256,
    depth: int = 6,
    cond_dim: int = 128,
    learning_rate: float = 3e-4,
    batch_size: int = 512,
    n_steps: int = 5_000,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    n_eval: int = 500,
    n_posterior_samples: int = 200,
    n_sde_steps: int = 100,
    seed: int = 0,
):
    """Train an MLP score network and evaluate fiber orientation error."""
    from dmipy_jax.inference.score_posterior import (
        MLPScoreNet, VPSchedule, train_score_posterior, sample_posterior,
    )

    key = jax.random.key(seed)
    k_net, k_train, k_eval = jax.random.split(key, 3)

    # Build acquisition and simulators
    acq = make_hcp_acquisition()
    sim_train = build_simulator(acq, snr=30.0, snr_range=(10.0, 50.0))
    sim_test = build_simulator(acq, snr=30.0)

    # Normalisation bounds
    lows = jnp.array([sim_train.parameter_ranges[n][0]
                       for n in sim_train.parameter_names])
    highs = jnp.array([sim_train.parameter_ranges[n][1]
                        for n in sim_train.parameter_names])
    spans = jnp.maximum(highs - lows, 1e-12)
    b0_mask = sim_train.acquisition.bvalues < 100e6

    def sim_fn(k, theta_norm):
        k1, k2 = jax.random.split(k)
        theta = theta_norm * spans + lows
        signal = sim_train.simulate(k1, theta)
        noisy = sim_train.add_noise(k2, signal)
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean
        return noisy

    def prior_fn(k, n):
        theta = sim_train.prior_sampler(k, n)
        return (theta - lows) / spans

    # Build network and schedule
    score_net = MLPScoreNet(
        k_net,
        param_dim=sim_train.theta_dim,
        signal_dim=sim_train.signal_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        cond_dim=cond_dim,
    )
    schedule = VPSchedule(beta_min=beta_min, beta_max=beta_max)

    # Train
    t0 = time.time()
    score_net, losses = train_score_posterior(
        k_train, score_net,
        simulator_fn=sim_fn, prior_fn=prior_fn, schedule=schedule,
        num_steps=n_steps, batch_size=batch_size,
        learning_rate=learning_rate,
        print_every=max(1, n_steps // 5),
    )
    train_time = time.time() - t0

    # Evaluate
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, n_eval)

    @eqx.filter_jit
    def predict(net, sched, x):
        samples_norm = sample_posterior(
            jax.random.key(0), net, x, sched,
            n_samples=n_posterior_samples, n_steps=n_sde_steps,
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

    preds = jax.vmap(predict, in_axes=(None, None, 0))(
        score_net, schedule, signals_test
    )
    preds_np = np.asarray(preds)
    theta_np = np.asarray(theta_test)

    # Compute metrics
    errors1 = compute_fiber_errors(theta_np, preds_np, n_eval)

    result = ExperimentResult(
        commit=get_commit_hash(),
        model_name="Ball2Stick",
        architecture=f"MLPScore_h{hidden_dim}_d{depth}",
        dataset="synthetic_hcp",
        fiber1_median_deg=float(np.median(errors1)),
        fiber1_mean_deg=float(np.mean(errors1)),
        d_stick_r=safe_pearson_r(theta_np[:, 1], preds_np[:, 1]),
        f1_r=safe_pearson_r(theta_np[:, 2], preds_np[:, 2]),
        final_loss=float(losses[-1]),
        train_time_s=train_time,
        n_steps=n_steps,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )

    print(f"Fiber 1 median error: {result.fiber1_median_deg:.1f} deg")
    print(f"d_stick r={result.d_stick_r:.3f}  f1 r={result.f1_r:.3f}")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Time: {train_time:.0f}s")

    print_result(result)
    log_result(result)

    return result


if __name__ == "__main__":
    run_experiment()
