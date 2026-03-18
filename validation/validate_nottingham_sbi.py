#!/usr/bin/env python3
"""
Validation 2: Nottingham SBI replication — Ball+2Stick NPE.

Trains an SBI model (MDN) for the Ball+2Stick model and compares
posterior means against ground truth on synthetic data.

Targets:
  - r > 0.95 for diffusivity and volume fractions
  - < 3 degree median orientation error for primary fiber

Usage::

    python validation/validate_nottingham_sbi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.inference.mdn import MixtureDensityNetwork
import equinox as eqx


def make_hcp_like_acquisition():
    """90-direction multi-shell mimicking HCP."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (6, 1))
    v1 = rand_vecs(k1, 30)
    v2 = rand_vecs(k2, 30)
    v3 = rand_vecs(k3, 24)

    bvals = jnp.concatenate([
        jnp.zeros(6),
        jnp.full(30, 1e9),
        jnp.full(30, 2e9),
        jnp.full(24, 3e9),
    ])
    bvecs = jnp.concatenate([v0, v1, v2, v3], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def build_ball_2stick_simulator(acq, snr=30.0):
    """Ball + 2 Stick model.

    Parameters: [d_ball, d_stick, f1, f2, theta1, phi1, theta2, phi2]
    (8-D posterior)
    """
    def forward_fn(params, acq):
        d_ball = params[0]
        d_stick = params[1]
        f1 = params[2]
        f2 = params[3]
        theta1, phi1 = params[4], params[5]
        theta2, phi2 = params[6], params[7]
        f_ball = jnp.clip(1.0 - f1 - f2, 0.0, 1.0)

        mu1 = jnp.array([jnp.sin(theta1) * jnp.cos(phi1),
                          jnp.sin(theta1) * jnp.sin(phi1),
                          jnp.cos(theta1)])
        mu2 = jnp.array([jnp.sin(theta2) * jnp.cos(phi2),
                          jnp.sin(theta2) * jnp.sin(phi2),
                          jnp.cos(theta2)])

        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2

        s1 = jnp.exp(-acq.bvalues * d_stick * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_stick * cos2 ** 2)
        s_ball = jnp.exp(-acq.bvalues * d_ball)

        return f1 * s1 + f2 * s2 + f_ball * s_ball

    def prior_fn(key, n):
        keys = jax.random.split(key, 8)
        d_ball = jax.random.uniform(keys[0], (n,), minval=1.0e-9, maxval=3.5e-9)
        d_stick = jax.random.uniform(keys[1], (n,), minval=0.5e-9, maxval=2.5e-9)
        # Ordered fractions
        f1_raw = jax.random.uniform(keys[2], (n,), minval=0.1, maxval=0.8)
        f2_raw = jax.random.uniform(keys[3], (n,), minval=0.05, maxval=0.5)
        total = f1_raw + f2_raw
        # Ensure sum <= 0.95
        scale = jnp.minimum(1.0, 0.95 / total)
        f1 = f1_raw * scale
        f2 = f2_raw * scale
        theta1 = jax.random.uniform(keys[4], (n,), minval=0.0, maxval=jnp.pi)
        phi1 = jax.random.uniform(keys[5], (n,), minval=-jnp.pi, maxval=jnp.pi)
        theta2 = jax.random.uniform(keys[6], (n,), minval=0.0, maxval=jnp.pi)
        phi2 = jax.random.uniform(keys[7], (n,), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.stack([d_ball, d_stick, f1, f2, theta1, phi1, theta2, phi2], axis=-1)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_ball", "d_stick", "f1", "f2",
                          "theta1", "phi1", "theta2", "phi2"],
        parameter_ranges={
            "d_ball": (1.0e-9, 3.5e-9),
            "d_stick": (0.5e-9, 2.5e-9),
            "f1": (0.1, 0.8), "f2": (0.05, 0.5),
            "theta1": (0.0, float(jnp.pi)), "phi1": (-float(jnp.pi), float(jnp.pi)),
            "theta2": (0.0, float(jnp.pi)), "phi2": (-float(jnp.pi), float(jnp.pi)),
        },
        acquisition=acq,
        noise_type="rician",
        snr=snr,
        prior_sampler_fn=prior_fn,
    )


def angular_error_deg(theta_t, phi_t, theta_r, phi_r):
    """Angle between two orientations (handles antipodal symmetry)."""
    mu_t = np.array([np.sin(theta_t) * np.cos(phi_t),
                      np.sin(theta_t) * np.sin(phi_t),
                      np.cos(theta_t)])
    mu_r = np.array([np.sin(theta_r) * np.cos(phi_r),
                      np.sin(theta_r) * np.sin(phi_r),
                      np.cos(theta_r)])
    dot = np.abs(np.dot(mu_t, mu_r))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def main():
    print("=" * 70)
    print("Nottingham SBI Replication: Ball + 2 Stick NPE")
    print("=" * 70)

    acq = make_hcp_like_acquisition()
    sim = build_ball_2stick_simulator(acq, snr=30.0)

    config = SBIPipelineConfig(
        model_name="Ball2Stick",
        parameter_names=sim.parameter_names,
        parameter_ranges=sim.parameter_ranges,
        acquisition={"bvalues": acq.bvalues.tolist()},
        inference_mode="mdn",
        n_components=10,
        hidden_dim=256,
        depth=4,
        learning_rate=5e-4,
        batch_size=512,
        n_steps=10_000,
        snr=30.0,
    )

    print(f"\nTraining MDN ({config.n_steps} steps)...")
    t0 = time.time()
    model, losses = train_sbi(config, sim, print_every=2000)
    print(f"Training time: {time.time() - t0:.1f} s")

    # Test set
    print("\nEvaluating on 2000 test samples...")
    key_test = jax.random.PRNGKey(999)
    theta_test, signals_test = sim.sample_and_simulate(key_test, 2000)

    @eqx.filter_jit
    def predict_mean(mdl, x):
        logits_pi, mu, log_sigma = mdl(x)
        weights = jax.nn.softmax(logits_pi)
        return jnp.sum(weights[:, None] * mu, axis=0)

    preds = jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_test)
    preds_np = np.asarray(preds)
    theta_np = np.asarray(theta_test)

    # Correlations for scalar params
    print("\nCorrelations (Pearson r):")
    scalar_names = ["d_ball", "d_stick", "f1", "f2"]
    for i, name in enumerate(scalar_names):
        r = np.corrcoef(theta_np[:, i], preds_np[:, i])[0, 1]
        rmse = np.sqrt(np.mean((theta_np[:, i] - preds_np[:, i]) ** 2))
        print(f"  {name:>10s}: r={r:.4f}  RMSE={rmse:.2e}")

    # Orientation errors
    orient_errors_1 = []
    orient_errors_2 = []
    for j in range(len(theta_np)):
        e1 = angular_error_deg(theta_np[j, 4], theta_np[j, 5],
                                preds_np[j, 4], preds_np[j, 5])
        e2 = angular_error_deg(theta_np[j, 6], theta_np[j, 7],
                                preds_np[j, 6], preds_np[j, 7])
        orient_errors_1.append(e1)
        orient_errors_2.append(e2)

    orient_errors_1 = np.array(orient_errors_1)
    orient_errors_2 = np.array(orient_errors_2)
    print(f"\nOrientation errors:")
    print(f"  Fiber 1: median={np.median(orient_errors_1):.1f} deg, "
          f"mean={np.mean(orient_errors_1):.1f} deg")
    print(f"  Fiber 2: median={np.median(orient_errors_2):.1f} deg, "
          f"mean={np.mean(orient_errors_2):.1f} deg")

    # Summary vs targets
    print("\n" + "=" * 70)
    print("Target Comparison")
    print("=" * 70)
    d_ball_r = np.corrcoef(theta_np[:, 0], preds_np[:, 0])[0, 1]
    f1_r = np.corrcoef(theta_np[:, 2], preds_np[:, 2])[0, 1]
    print(f"  d_ball correlation:     r={d_ball_r:.4f}  (target: >0.95)")
    print(f"  f1 correlation:         r={f1_r:.4f}  (target: >0.95)")
    print(f"  Fiber 1 median error:   {np.median(orient_errors_1):.1f} deg  (target: <3 deg)")

    np.savez(
        "validation/nottingham_sbi_results.npz",
        theta_test=theta_np,
        predictions=preds_np,
        orient_errors_1=orient_errors_1,
        orient_errors_2=orient_errors_2,
        losses=np.array(losses),
    )
    print("\nResults saved to validation/nottingham_sbi_results.npz")


if __name__ == "__main__":
    main()
