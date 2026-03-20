#!/usr/bin/env python3
"""
Validation: Simulation-Based Calibration for DTI MDN.

Trains a DTI MDN, then runs SBC to verify posterior calibration.
Well-calibrated posteriors should show:
  - Uniform rank histograms
  - KS test p-values > 0.05
  - 90% coverage close to 90% (+/- 5%)

Usage::

    python validation/validate_sbc.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.sbc import SBCDiagnostic


def make_dti_simulator():
    """Build a simple DTI simulator (FA, MD_scaled)."""
    key = jax.random.PRNGKey(0)
    n_dirs = 32
    bvecs = jax.random.normal(key, (n_dirs, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    # Include a b0 measurement so b0-normalisation works
    bvals = jnp.concatenate([jnp.zeros(1), jnp.full(n_dirs, 1e9)])
    bvecs_with_b0 = jnp.concatenate(
        [jnp.array([[1.0, 0.0, 0.0]]), bvecs], axis=0
    )
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs_with_b0)

    def forward_fn(params, acq):
        """Simplified DTI: params = [FA_scaled, MD_scaled]."""
        fa = jnp.clip(params[0], 0.0, 1.0)
        md = jnp.clip(params[1], 0.1, 3.0) * 1e-9  # un-scale

        l1 = md * (1 + 2.0 / 3.0 * fa)
        l23 = md * (1 - 1.0 / 3.0 * fa)

        d_avg = (l1 + 2 * l23) / 3.0
        return jnp.exp(-acq.bvalues * d_avg)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["FA", "MD_scaled"],
        parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
        acquisition=acq,
        noise_type="rician",
        snr=30.0,
    ), acq


def main():
    print("=" * 60)
    print("SBC Validation: DTI MDN")
    print("=" * 60)

    # 1. Build simulator
    sim, acq = make_dti_simulator()
    print(f"Simulator: {sim.theta_dim} params, {sim.signal_dim} measurements")

    # 2. Train MDN
    config = SBIPipelineConfig(
        model_name="DTI_SBC_validation",
        parameter_names=["FA", "MD_scaled"],
        parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
        acquisition={"bvalues": acq.bvalues.tolist()},
        inference_mode="mdn",
        n_components=8,
        hidden_dim=128,
        depth=3,
        learning_rate=1e-3,
        batch_size=256,
        n_steps=3000,
        snr=30.0,
    )

    key = jax.random.PRNGKey(42)
    print("\nTraining MDN...")
    t0 = time.time()
    model, losses = train_sbi(config, sim, key=key, print_every=500)
    t_train = time.time() - t0
    print(f"Training complete in {t_train:.1f}s.  Final loss: {losses[-1]:.4f}")

    # 3. Run SBC
    n_sbc = 500
    n_post = 200
    print(f"\nRunning SBC: {n_sbc} samples, {n_post} posterior draws each...")
    sbc = SBCDiagnostic(model, sim, n_posterior_samples=n_post)
    t0 = time.time()
    result = sbc.run(jax.random.PRNGKey(123), n_sbc_samples=n_sbc)
    t_sbc = time.time() - t0
    print(f"SBC complete in {t_sbc:.1f}s")

    # 4. Print summary
    print("\n" + "=" * 60)
    print("SBC Results")
    print("=" * 60)
    result.print_summary()

    # 5. Check pass criteria
    ks = result.ks_test()
    cov90 = result.coverage_at(0.9)
    all_pass = True
    for d, name in enumerate(result.parameter_names):
        if ks[name]["pvalue"] < 0.05:
            print(f"  WARNING: {name} KS test FAILED (p={ks[name]['pvalue']:.4f})")
            all_pass = False
        if abs(cov90[d] - 0.9) > 0.10:
            print(f"  WARNING: {name} coverage@90%={cov90[d]:.3f} far from 0.9")
            all_pass = False

    print()
    if all_pass:
        print("PASS: All SBC checks passed.")
    else:
        print("WARN: Some SBC checks did not pass -- review diagnostics.")

    # 6. Save rank histogram plot
    save_path = "validation/sbc_rank_histograms.png"
    result.plot_rank_histograms(save_path=save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
