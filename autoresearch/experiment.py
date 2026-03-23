#!/usr/bin/env python3
"""Baseline experiment: Spline NPE (flow-based) posterior for Ball + 2-Stick.

This is the root experiment for agentsciml evolutionary search.
It trains a masked autoregressive flow with rational-quadratic spline
transforms (via FlowJAX) and evaluates fiber orientation error on
synthetic HCP-like data.

Architecture from the 3.2-degree result:
  - FlowJAX masked_autoregressive_flow with RQS transformer
  - 10 flow layers, hidden_dim=128, 8 spline knots
  - Trained on normalised [0,1] parameters with b0-normalised signals
  - Posterior: 500 samples -> clip -> unit-normalise orientations -> median
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
    hidden_dim: int = 128,
    num_layers: int = 10,
    knots: int = 8,
    learning_rate: float = 3e-4,
    batch_size: int = 512,
    n_steps: int = 50_000,
    n_eval: int = 500,
    n_posterior_samples: int = 500,
    seed: int = 0,
):
    """Train a spline NPE flow and evaluate fiber orientation error."""
    from dmipy_jax.inference.trainer import create_trainer, train_loop

    key = jax.random.key(seed)
    k_flow, k_train, k_eval = jax.random.split(key, 3)

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

    # Build flow and optimizer
    flow, optimizer = create_trainer(
        flow_key=k_flow,
        theta_dim=sim_train.theta_dim,
        signal_dim=sim_train.signal_dim,
        simulator=sim_fn,
        prior_sampler=prior_fn,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        flow_type="spline",
        knots=knots,
    )

    # Train
    t0 = time.time()
    flow, losses = train_loop(
        flow,
        optimizer,
        simulator=sim_fn,
        prior_sampler=prior_fn,
        key=k_train,
        num_steps=n_steps,
        batch_size=batch_size,
        noise_std=0.0,
        print_every=max(1, n_steps // 5),
    )
    train_time = time.time() - t0

    # Evaluate
    theta_test, signals_test = sim_test.sample_and_simulate(k_eval, n_eval)

    @eqx.filter_jit
    def predict(flow_model, x):
        samples_norm = flow_model.sample(
            jax.random.key(0), (n_posterior_samples,), condition=x,
        )
        samples = samples_norm * spans + lows
        samples = jnp.clip(samples, lows, highs)
        # Normalise orientation vectors to unit sphere
        mu1 = samples[:, 4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1, axis=-1, keepdims=True), 1e-8)
        mu2 = samples[:, 7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2, axis=-1, keepdims=True), 1e-8)
        samples = samples.at[:, 4:7].set(mu1).at[:, 7:10].set(mu2)
        return jnp.median(samples, axis=0)

    preds = jax.vmap(predict, in_axes=(None, 0))(flow, signals_test)
    preds_np = np.asarray(preds)
    theta_np = np.asarray(theta_test)

    # Compute metrics
    errors1 = compute_fiber_errors(theta_np, preds_np, n_eval)

    result = ExperimentResult(
        commit=get_commit_hash(),
        model_name="Ball2Stick",
        architecture=f"SplineNPE_L{num_layers}_h{hidden_dim}_K{knots}",
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
