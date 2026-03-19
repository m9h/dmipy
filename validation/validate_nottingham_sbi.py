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
    """Ball + 2 Stick model with Cartesian orientation vectors.

    Parameters: [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
    (10-D posterior)

    Orientations are unit vectors canonicalized to the z>=0 hemisphere
    to remove antipodal ambiguity.  This gives the MDN a smooth,
    discontinuity-free target space unlike (theta, phi) angles.
    """
    def forward_fn(params, acq):
        d_ball = params[0]
        d_stick = params[1]
        f1 = params[2]
        f2 = params[3]
        f_ball = jnp.clip(1.0 - f1 - f2, 0.0, 1.0)

        mu1 = params[4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1), 1e-8)
        mu2 = params[7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2), 1e-8)

        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2

        s1 = jnp.exp(-acq.bvalues * d_stick * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_stick * cos2 ** 2)
        s_ball = jnp.exp(-acq.bvalues * d_ball)

        return f1 * s1 + f2 * s2 + f_ball * s_ball

    def _sample_hemisphere(key, n):
        """Sample unit vectors on the z>=0 hemisphere."""
        z = jax.random.normal(key, (n, 3))
        z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        # Canonicalize: flip to z>=0 hemisphere
        z = z * jnp.sign(z[:, 2:3] + 1e-8)
        return z

    def prior_fn(key, n):
        keys = jax.random.split(key, 6)
        d_ball = jax.random.uniform(keys[0], (n,), minval=1.0e-9, maxval=3.5e-9)
        d_stick = jax.random.uniform(keys[1], (n,), minval=0.5e-9, maxval=2.5e-9)
        f1_raw = jax.random.uniform(keys[2], (n,), minval=0.1, maxval=0.8)
        f2_raw = jax.random.uniform(keys[3], (n,), minval=0.05, maxval=0.5)
        total = f1_raw + f2_raw
        scale = jnp.minimum(1.0, 0.95 / total)
        f1 = f1_raw * scale
        f2 = f2_raw * scale
        mu1 = _sample_hemisphere(keys[4], n)  # (n, 3)
        mu2 = _sample_hemisphere(keys[5], n)  # (n, 3)
        return jnp.concatenate([
            d_ball[:, None], d_stick[:, None],
            f1[:, None], f2[:, None],
            mu1, mu2,
        ], axis=-1)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_ball", "d_stick", "f1", "f2",
                          "mu1x", "mu1y", "mu1z", "mu2x", "mu2y", "mu2z"],
        parameter_ranges={
            "d_ball": (1.0e-9, 3.5e-9),
            "d_stick": (0.5e-9, 2.5e-9),
            "f1": (0.1, 0.8), "f2": (0.05, 0.5),
            "mu1x": (-1.0, 1.0), "mu1y": (-1.0, 1.0), "mu1z": (0.0, 1.0),
            "mu2x": (-1.0, 1.0), "mu2y": (-1.0, 1.0), "mu2z": (0.0, 1.0),
        },
        acquisition=acq,
        noise_type="rician",
        snr=snr,
        prior_sampler_fn=prior_fn,
    )


def angular_error_deg_cartesian(mu_t, mu_r):
    """Angle between two unit vectors (handles antipodal symmetry)."""
    dot = np.abs(np.dot(mu_t, mu_r))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def main():
    print("=" * 70)
    print("Nottingham SBI Replication: Ball + 2 Stick NPE")
    print("=" * 70)

    acq = make_hcp_like_acquisition()
    sim = build_ball_2stick_simulator(acq, snr=30.0)

    import sys
    use_flow = "--flow" in sys.argv

    if use_flow:
        config = SBIPipelineConfig(
            model_name="Ball2Stick",
            parameter_names=sim.parameter_names,
            parameter_ranges=sim.parameter_ranges,
            acquisition={"bvalues": acq.bvalues.tolist()},
            inference_mode="flow",
            hidden_dim=128,
            depth=6,
            learning_rate=3e-4,
            batch_size=512,
            n_steps=30_000,
            snr=30.0,
        )
    else:
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
            n_steps=30_000,
            snr=30.0,
        )

    mode_str = "Flow (NPE)" if use_flow else "MDN"
    print(f"\nTraining {mode_str} ({config.n_steps} steps)...")
    t0 = time.time()
    model, losses = train_sbi(config, sim, print_every=5000)
    print(f"Training time: {time.time() - t0:.1f} s")

    # Test set
    print("\nEvaluating on 2000 test samples...")
    key_test = jax.random.PRNGKey(999)
    theta_test, signals_test = sim.sample_and_simulate(key_test, 2000)

    if use_flow:
        @eqx.filter_jit
        def predict_mean_flow(mdl, x):
            key = jax.random.key(0)
            samples = mdl.sample(key, (200,), condition=x)
            return jnp.mean(samples, axis=0)
        preds = jax.vmap(predict_mean_flow, in_axes=(None, 0))(model, signals_test)
    else:
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

    # Orientation errors — using Cartesian unit vectors directly
    orient_errors_1 = []
    orient_errors_2 = []
    for j in range(len(theta_np)):
        mu1_true = theta_np[j, 4:7]
        mu1_pred = preds_np[j, 4:7]
        mu1_pred = mu1_pred / max(np.linalg.norm(mu1_pred), 1e-8)
        mu2_true = theta_np[j, 7:10]
        mu2_pred = preds_np[j, 7:10]
        mu2_pred = mu2_pred / max(np.linalg.norm(mu2_pred), 1e-8)

        # Try both fiber assignments (label-switching symmetry)
        e11 = angular_error_deg_cartesian(mu1_true, mu1_pred)
        e22 = angular_error_deg_cartesian(mu2_true, mu2_pred)
        e12 = angular_error_deg_cartesian(mu1_true, mu2_pred)
        e21 = angular_error_deg_cartesian(mu2_true, mu1_pred)
        if e11 + e22 <= e12 + e21:
            orient_errors_1.append(e11)
            orient_errors_2.append(e22)
        else:
            orient_errors_1.append(e12)
            orient_errors_2.append(e21)

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
