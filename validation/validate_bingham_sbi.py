#!/usr/bin/env python3
"""
Validation 4: Bingham-NODDI SBI — anisotropic dispersion posteriors.

Trains the Bingham-NODDI SBI model and verifies that the posterior
correctly recovers kappa1 != kappa2 (anisotropic dispersion), which
no existing SBI tool produces.

Key test: when ground truth has kappa1 >> kappa2 (or vice versa),
the posterior mean should reflect the asymmetry.

Usage::

    python validation/validate_bingham_sbi.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.models.bingham_noddi import (
    build_bingham_noddi_simulator,
    get_bingham_noddi_config,
)
from dmipy_jax.inference.mdn import MixtureDensityNetwork
import equinox as eqx


def make_multishell():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (2, 1))
    v1 = rand_vecs(k1, 32)
    v2 = rand_vecs(k2, 32)
    bvals = jnp.concatenate([jnp.zeros(2), jnp.full(32, 700e6), jnp.full(32, 2000e6)])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def main():
    print("=" * 70)
    print("Bingham-NODDI SBI Validation: Anisotropic Dispersion Recovery")
    print("=" * 70)

    acq = make_multishell()
    sim = build_bingham_noddi_simulator(acq, snr=30.0)
    config = get_bingham_noddi_config(acq, n_steps=5000)

    print(f"\nTraining Bingham-NODDI MDN ({config.n_steps} steps)...")
    t0 = time.time()
    model, losses = train_sbi(config, sim, print_every=1000)
    print(f"Training time: {time.time() - t0:.1f} s")
    print(f"Final loss: {losses[-1]:.4f}")

    # Test: generate samples with known anisotropic dispersion
    print("\nGenerating test cases with controlled kappa asymmetry...")

    @eqx.filter_jit
    def predict_mean(mdl, x):
        logits_pi, mu, log_sigma = mdl(x)
        weights = jax.nn.softmax(logits_pi)
        return jnp.sum(weights[:, None] * mu, axis=0)

    n_test = 1000
    key_test = jax.random.PRNGKey(42)
    theta_test, signals_test = sim.sample_and_simulate(key_test, n_test)
    preds = np.asarray(jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_test))
    theta_np = np.asarray(theta_test)

    # Parameter indices: [f_intra, f_iso, kappa1, kappa2, mu_theta]
    param_names = config.parameter_names

    print("\nOverall recovery (all test samples):")
    for i, name in enumerate(param_names):
        r = np.corrcoef(theta_np[:, i], preds[:, i])[0, 1]
        rmse = np.sqrt(np.mean((theta_np[:, i] - preds[:, i]) ** 2))
        bias = np.mean(preds[:, i] - theta_np[:, i])
        print(f"  {name:>10s}: r={r:.4f}  RMSE={rmse:.3f}  bias={bias:+.3f}")

    # Critical test: kappa asymmetry recovery
    print("\n--- Kappa Asymmetry Test ---")
    k1_true = theta_np[:, 2]
    k2_true = theta_np[:, 3]
    k1_pred = preds[:, 2]
    k2_pred = preds[:, 3]

    # Classify: anisotropic if |log(k1/k2)| > 0.5
    log_ratio_true = np.log(k1_true / k2_true)
    log_ratio_pred = np.log(np.maximum(k1_pred, 0.1) / np.maximum(k2_pred, 0.1))

    aniso_mask = np.abs(log_ratio_true) > 0.5
    n_aniso = np.sum(aniso_mask)
    print(f"  Anisotropic samples (|log(k1/k2)| > 0.5): {n_aniso}/{n_test}")

    if n_aniso > 10:
        # Check sign agreement of log-ratio
        sign_agree = np.mean(
            np.sign(log_ratio_true[aniso_mask]) == np.sign(log_ratio_pred[aniso_mask])
        )
        r_ratio = np.corrcoef(log_ratio_true[aniso_mask], log_ratio_pred[aniso_mask])[0, 1]
        print(f"  Sign agreement (k1>k2 vs k1<k2):  {sign_agree:.1%}")
        print(f"  log(k1/k2) correlation:            r={r_ratio:.4f}")
        print(f"  Target: sign agreement > 60%, r > 0.3")

    # Relative error
    for i, name in enumerate(param_names[:4]):  # skip mu_theta
        rel_err = np.mean(np.abs(preds[:, i] - theta_np[:, i]) / (np.abs(theta_np[:, i]) + 1e-8))
        print(f"  {name:>10s} mean relative error: {rel_err:.1%}  (target: <10%)")

    np.savez(
        "validation/bingham_sbi_results.npz",
        theta_test=theta_np,
        predictions=preds,
        losses=np.array(losses),
        param_names=param_names,
    )
    print("\nResults saved to validation/bingham_sbi_results.npz")


if __name__ == "__main__":
    main()
