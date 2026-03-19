#!/usr/bin/env python3
"""
Validation 3: Alicante SBIDTI replication — DTI NPE with minimal acquisitions.

Trains SBI models and compares three acquisition strategies:
  1. Full scheme (33 measurements: 1 b0 + 32 directions)
  2. Fisher-optimal minimal scheme (7 measurements via D-optimal design)
  3. Random minimal scheme (7 measurements, electrostatic-repulsion equivalent)

Demonstrates that Fisher-optimised minimal acquisitions retain accuracy
close to the full scheme (Alicante's key finding).

Target: SSIM > 0.9 on FA/MD for the full scheme.

Usage::

    python validation/validate_alicante_sbi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.library.derived_metrics import tensor_to_fa_md
from dmipy_jax.design.oed import d_optimality_loss, optimize_protocol
from dmipy_jax.inference.mdn import MixtureDensityNetwork
import equinox as eqx


def make_acquisition(n_dirs, b_value=1e9, n_b0=1, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    z = jax.random.normal(key, (n_dirs, 3))
    bvecs = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n_b0, 1))
    bvals = jnp.concatenate([jnp.zeros(n_b0), jnp.full(n_dirs, b_value)])
    all_vecs = jnp.concatenate([v0, bvecs], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=all_vecs)


def dti_forward_fn(params, acq):
    """DTI forward model: params = [l1, l2, l3, theta, phi]."""
    l1, l2, l3, theta, phi = params[0], params[1], params[2], params[3], params[4]
    mu = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                     jnp.sin(theta) * jnp.sin(phi),
                     jnp.cos(theta)])
    y_axis = jnp.array([0.0, 1.0, 0.0])
    e2 = jnp.cross(mu, y_axis)
    e2_norm = jnp.linalg.norm(e2)
    e2 = jnp.where(e2_norm > 1e-6, e2 / e2_norm, jnp.array([1.0, 0.0, 0.0]))
    e3 = jnp.cross(mu, e2)
    D = l1 * jnp.outer(mu, mu) + l2 * jnp.outer(e2, e2) + l3 * jnp.outer(e3, e3)
    g = acq.gradient_directions
    gDg = jnp.sum((g @ D) * g, axis=-1)
    return jnp.exp(-acq.bvalues * gDg)


def build_dti_simulator(acq, snr=30.0):
    def prior_fn(key, n):
        keys = jax.random.split(key, 5)
        l1 = jax.random.uniform(keys[0], (n,), minval=0.5e-9, maxval=3.0e-9)
        l2 = jax.random.uniform(keys[1], (n,), minval=0.1e-9, maxval=l1)
        l3 = jax.random.uniform(keys[2], (n,), minval=0.1e-9, maxval=l2)
        theta = jax.random.uniform(keys[3], (n,), minval=0.0, maxval=jnp.pi)
        phi = jax.random.uniform(keys[4], (n,), minval=-jnp.pi, maxval=jnp.pi)
        return jnp.stack([l1, l2, l3, theta, phi], axis=-1)

    return ModelSimulator(
        forward_fn=dti_forward_fn,
        parameter_names=["l1", "l2", "l3", "theta", "phi"],
        parameter_ranges={
            "l1": (0.5e-9, 3.0e-9), "l2": (0.1e-9, 3.0e-9),
            "l3": (0.1e-9, 3.0e-9),
            "theta": (0.0, float(jnp.pi)),
            "phi": (-float(jnp.pi), float(jnp.pi)),
        },
        acquisition=acq, noise_type="rician", snr=snr,
        prior_sampler_fn=prior_fn,
    )


def optimize_minimal_acquisition(n_dirs=6, b_value=1e9, n_steps=100):
    """Use D-optimal design to find the best 6 gradient directions."""
    # Start from random directions
    initial_acq = make_acquisition(n_dirs, b_value=b_value, n_b0=1,
                                   key=jax.random.PRNGKey(99))

    # Target parameters for Fisher information
    target_params = {
        "l1": jnp.array(1.7e-9),
        "l2": jnp.array(0.5e-9),
        "l3": jnp.array(0.3e-9),
        "theta": jnp.array(jnp.pi / 4),
        "phi": jnp.array(0.0),
    }

    # Model function for OED (kwargs-based)
    def model_for_oed(bvalues, gradient_directions, l1, l2, l3, theta, phi, **kw):
        mu = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                         jnp.sin(theta) * jnp.sin(phi),
                         jnp.cos(theta)])
        y_axis = jnp.array([0.0, 1.0, 0.0])
        e2 = jnp.cross(mu, y_axis)
        e2_norm = jnp.linalg.norm(e2)
        e2 = jnp.where(e2_norm > 1e-6, e2 / e2_norm, jnp.array([1.0, 0.0, 0.0]))
        e3 = jnp.cross(mu, e2)
        D = l1 * jnp.outer(mu, mu) + l2 * jnp.outer(e2, e2) + l3 * jnp.outer(e3, e3)
        gDg = jnp.sum((gradient_directions @ D) * gradient_directions, axis=-1)
        return jnp.exp(-bvalues * gDg)

    print("  Optimising gradient directions (D-optimal)...")
    optimised_acq, history = optimize_protocol(
        initial_acq, model_for_oed, target_params,
        n_steps=n_steps, learning_rate=0.05, sigma=1.0 / 30.0,
        return_history=True,
    )
    print(f"  D-optimality loss: {history['loss'][0]:.2f} → {history['loss'][-1]:.2f}")
    return optimised_acq


def ssim_1d(x, y):
    """Simplified SSIM for 1-D arrays."""
    c1 = (0.01 * (np.max(x) - np.min(x))) ** 2
    c2 = (0.03 * (np.max(x) - np.min(x))) ** 2
    mu_x, mu_y = np.mean(x), np.mean(y)
    sig_x, sig_y = np.std(x), np.std(y)
    sig_xy = np.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x ** 2 + sig_y ** 2 + c2)
    return num / den


def train_and_evaluate(name, acq, theta_test_shared, n_steps=5000):
    """Train MDN on a given acquisition and evaluate FA/MD."""
    sim = build_dti_simulator(acq, snr=30.0)
    config = SBIPipelineConfig(
        model_name=name, parameter_names=sim.parameter_names,
        acquisition={"bvalues": acq.bvalues.tolist()},
        inference_mode="mdn", n_components=8, hidden_dim=128, depth=3,
        n_steps=n_steps, batch_size=256, snr=30.0,
    )
    model, losses = train_sbi(config, sim, print_every=n_steps)

    # Generate test signals from shared theta using this acquisition
    signals = jax.vmap(lambda t: dti_forward_fn(t, acq))(theta_test_shared)
    # Add noise
    key = jax.random.PRNGKey(777)
    k1, k2 = jax.random.split(key)
    sigma = 1.0 / 30.0
    n1 = jax.random.normal(k1, signals.shape) * sigma
    n2 = jax.random.normal(k2, signals.shape) * sigma
    signals_noisy = jnp.sqrt((signals + n1) ** 2 + n2 ** 2)

    @eqx.filter_jit
    def predict_mean(mdl, x):
        logits_pi, mu, log_sigma = mdl(x)
        weights = jax.nn.softmax(logits_pi)
        return jnp.sum(weights[:, None] * mu, axis=0)

    preds = np.asarray(jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_noisy))
    return preds, losses


def main():
    print("=" * 70)
    print("Alicante SBI Replication: DTI NPE — Full vs Fisher-Optimal vs Random")
    print("=" * 70)

    N_STEPS = 15000

    # 1. Full acquisition
    acq_full = make_acquisition(32, b_value=1e9, n_b0=1)
    print(f"\nFull scheme: {acq_full.bvalues.shape[0]} measurements")

    # 2. Fisher-optimal minimal (D-optimal gradient directions)
    acq_fisher = optimize_minimal_acquisition(n_dirs=6, n_steps=500)
    print(f"Fisher-optimal minimal: {acq_fisher.bvalues.shape[0]} measurements")

    # 3. Random minimal (electrostatic-repulsion proxy)
    acq_random = make_acquisition(6, b_value=1e9, n_b0=1, key=jax.random.PRNGKey(123))
    print(f"Random minimal: {acq_random.bvalues.shape[0]} measurements")

    # Shared test set (ground truth parameters)
    sim_ref = build_dti_simulator(acq_full)
    key_test = jax.random.PRNGKey(999)
    theta_test = sim_ref.prior_sampler(key_test, 2000)
    theta_np = np.asarray(theta_test)

    true_eigs = theta_np[:, :3]
    fa_true, md_true = tensor_to_fa_md(jnp.array(true_eigs))
    fa_true, md_true = np.asarray(fa_true), np.asarray(md_true)

    # Train and evaluate each
    configs = [
        ("Full (33)", acq_full),
        ("Fisher-opt (7)", acq_fisher),
        ("Random (7)", acq_random),
    ]

    all_results = {}
    for name, acq in configs:
        print(f"\n--- Training: {name} ({N_STEPS} steps) ---")
        preds, losses = train_and_evaluate(name, acq, theta_test, n_steps=N_STEPS)
        pred_eigs = preds[:, :3]
        fa_pred, md_pred = tensor_to_fa_md(jnp.array(pred_eigs))
        fa_pred, md_pred = np.asarray(fa_pred), np.asarray(md_pred)
        all_results[name] = {
            "fa": fa_pred, "md": md_pred,
            "ssim_fa": ssim_1d(fa_true, fa_pred),
            "ssim_md": ssim_1d(md_true, md_pred),
            "r_fa": np.corrcoef(fa_true, fa_pred)[0, 1],
        }

    # Report
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"{'Scheme':>18s}  {'FA SSIM':>8s}  {'MD SSIM':>8s}  {'FA r':>8s}")
    for name, r in all_results.items():
        print(f"{name:>18s}  {r['ssim_fa']:>8.4f}  {r['ssim_md']:>8.4f}  {r['r_fa']:>8.4f}")

    print(f"\nTarget: Full scheme FA SSIM > 0.9")
    print(f"Expected: Fisher-optimal > Random for minimal schemes")

    save_dict = {"fa_true": fa_true, "md_true": md_true}
    for name, r in all_results.items():
        tag = name.replace(" ", "_").replace("(", "").replace(")", "")
        save_dict[f"fa_{tag}"] = r["fa"]
        save_dict[f"md_{tag}"] = r["md"]
        save_dict[f"ssim_fa_{tag}"] = r["ssim_fa"]
        save_dict[f"ssim_md_{tag}"] = r["ssim_md"]

    np.savez("validation/alicante_sbi_results.npz", **save_dict)
    print("Results saved to validation/alicante_sbi_results.npz")


if __name__ == "__main__":
    main()
