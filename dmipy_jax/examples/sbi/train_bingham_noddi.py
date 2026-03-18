#!/usr/bin/env python3
"""
Train a Bingham-NODDI SBI model using the unified pipeline.

Demonstrates Phase 5: Bingham-distributed NODDI with anisotropic
dispersion (kappa1 != kappa2).

Usage::

    python -m dmipy_jax.examples.sbi.train_bingham_noddi
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.checkpoint import save_checkpoint
from dmipy_jax.pipeline.models.bingham_noddi import (
    build_bingham_noddi_simulator,
    get_bingham_noddi_config,
)


def make_multishell_acquisition(key, n_dirs=32):
    """Create a 2-shell acquisition (b=700, 2000 s/mm^2)."""
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v1 = rand_vecs(k1, n_dirs)
    v2 = rand_vecs(k2, n_dirs)
    v0 = jnp.array([[1.0, 0.0, 0.0]] * 2)

    bvals = jnp.concatenate([
        jnp.zeros(2),
        jnp.full(n_dirs, 700e6),
        jnp.full(n_dirs, 2000e6),
    ])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def main():
    print(f"Device: {jax.devices()[0]}")

    key = jax.random.PRNGKey(42)
    k_acq, k_train = jax.random.split(key)

    # Build acquisition
    acq = make_multishell_acquisition(k_acq)
    print(f"Acquisition: {acq.bvalues.shape[0]} measurements")

    # Build simulator and config
    simulator = build_bingham_noddi_simulator(acq, snr=30.0)
    config = get_bingham_noddi_config(acq, n_steps=5000)

    # Train
    print("Starting Bingham-NODDI SBI training...")
    model, losses = train_sbi(config, simulator, key=k_train, print_every=500)

    # Save checkpoint
    save_checkpoint(model, config, "checkpoints/bingham_noddi")
    print("Checkpoint saved to checkpoints/bingham_noddi.*")

    # Validation
    print("Generating validation plots...")
    k_test = jax.random.PRNGKey(999)
    theta_test, signals_test = simulator.sample_and_simulate(k_test, 500)

    # Predict (posterior mean from MDN)
    from dmipy_jax.inference.mdn import MixtureDensityNetwork
    import equinox as eqx

    @eqx.filter_jit
    def predict_mean(mdl, x):
        logits_pi, mu, log_sigma = mdl(x)
        weights = jax.nn.softmax(logits_pi)
        return jnp.sum(weights[:, None] * mu, axis=0)

    preds = jax.vmap(predict_mean, in_axes=(None, 0))(model, signals_test)

    # Plot
    names = config.parameter_names
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    theta_np = np.asarray(theta_test)
    preds_np = np.asarray(preds)

    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.scatter(theta_np[:, i], preds_np[:, i], alpha=0.3, s=5)
        lo, hi = theta_np[:, i].min(), theta_np[:, i].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("bingham_noddi_validation.png", dpi=150)
    print("Saved bingham_noddi_validation.png")

    # Loss curve
    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("NLL Loss")
    plt.title("Bingham-NODDI SBI Training")
    plt.savefig("bingham_noddi_loss.png")
    print("Saved bingham_noddi_loss.png")


if __name__ == "__main__":
    main()
