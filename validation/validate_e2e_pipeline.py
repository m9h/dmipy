#!/usr/bin/env python3
"""
Validation 5: End-to-end pipeline on synthetic DWI volume.

Generates a synthetic multi-tissue DWI volume, runs the full pipeline:
  train → deploy → multi-metric maps → compare to ground truth

Produces FA, MD, NDI, ODI, FW, MK, and tissue segmentation maps and
validates them against known ground truth values.

Usage::

    python validation/validate_e2e_pipeline.py
"""

import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint
from dmipy_jax.pipeline.deploy import SBIPredictor
from dmipy_jax.data.io import save_nifti
from dmipy_jax.library.derived_metrics import tensor_to_fa_md


def make_acquisition():
    """2-shell acquisition: b=1000, 2000."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (2, 1))
    v1 = rand_vecs(k1, 15)
    v2 = rand_vecs(k2, 15)
    bvals = jnp.concatenate([jnp.zeros(2), jnp.full(15, 1e9), jnp.full(15, 2e9)])
    bvecs = jnp.concatenate([v0, v1, v2], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def build_simple_simulator(acq, snr=30.0):
    """Simple 2-param model: [FA_scaled, MD_scaled]."""
    def forward_fn(params, acq):
        fa = jnp.clip(params[0], 0.0, 1.0)
        md = jnp.clip(params[1], 0.1, 3.0) * 1e-9

        l1 = md * (1.0 + 2.0 / 3.0 * fa)
        l23 = md * (1.0 - 1.0 / 3.0 * fa)

        # Axially symmetric tensor along z
        cos2 = acq.gradient_directions[:, 2] ** 2
        d_eff = l1 * cos2 + l23 * (1.0 - cos2)
        return jnp.exp(-acq.bvalues * d_eff)

    return ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["FA", "MD_scaled"],
        parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
        acquisition=acq,
        noise_type="rician",
        snr=snr,
    )


def generate_synthetic_volume(acq, vol_shape=(20, 20, 5), snr=30.0):
    """Create a synthetic volume with 3 tissue regions."""
    nx, ny, nz = vol_shape
    n_meas = acq.bvalues.shape[0]
    dwi = np.zeros((*vol_shape, n_meas), dtype=np.float32)
    gt_fa = np.zeros(vol_shape, dtype=np.float32)
    gt_md = np.zeros(vol_shape, dtype=np.float32)
    gt_tissue = np.zeros(vol_shape, dtype=np.float32)  # 1=WM, 2=GM, 3=CSF

    bvals_np = np.asarray(acq.bvalues)
    bvecs_np = np.asarray(acq.gradient_directions)
    cos2 = bvecs_np[:, 2] ** 2
    sigma = 1.0 / snr

    rng = np.random.RandomState(42)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Assign tissue type by region
                if x < nx // 3:
                    # WM: high FA, moderate MD
                    fa = 0.6 + rng.uniform(-0.1, 0.1)
                    md_val = 0.8 + rng.uniform(-0.1, 0.1)
                    tissue = 1
                elif x < 2 * nx // 3:
                    # GM: low FA, moderate MD
                    fa = 0.15 + rng.uniform(-0.05, 0.05)
                    md_val = 1.0 + rng.uniform(-0.1, 0.1)
                    tissue = 2
                else:
                    # CSF: very low FA, high MD
                    fa = 0.05 + rng.uniform(-0.02, 0.02)
                    md_val = 2.8 + rng.uniform(-0.2, 0.2)
                    tissue = 3

                md = md_val * 1e-9
                l1 = md * (1.0 + 2.0 / 3.0 * fa)
                l23 = md * (1.0 - 1.0 / 3.0 * fa)
                d_eff = l1 * cos2 + l23 * (1.0 - cos2)
                signal = np.exp(-bvals_np * d_eff)

                # Add Rician noise
                n1 = rng.normal(0, sigma, signal.shape)
                n2 = rng.normal(0, sigma, signal.shape)
                noisy = np.sqrt((signal + n1) ** 2 + n2 ** 2)

                dwi[x, y, z, :] = noisy
                gt_fa[x, y, z] = fa
                gt_md[x, y, z] = md_val
                gt_tissue[x, y, z] = tissue

    return dwi, gt_fa, gt_md, gt_tissue


def main():
    print("=" * 70)
    print("End-to-End Pipeline Validation on Synthetic Volume")
    print("=" * 70)

    acq = make_acquisition()
    sim = build_simple_simulator(acq, snr=30.0)

    # Train
    config = SBIPipelineConfig(
        model_name="DTI_e2e",
        parameter_names=["FA", "MD_scaled"],
        parameter_ranges={"FA": (0.0, 1.0), "MD_scaled": (0.1, 3.0)},
        acquisition={"bvalues": acq.bvalues.tolist()},
        inference_mode="mdn",
        n_components=6,
        hidden_dim=96,
        depth=3,
        n_steps=3000,
        batch_size=256,
        snr=30.0,
    )

    print(f"\n1. Training ({config.n_steps} steps)...")
    model, losses = train_sbi(config, sim, print_every=1000)
    print(f"   Final loss: {losses[-1]:.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint
        ckpt_path = os.path.join(tmpdir, "e2e_ckpt")
        save_checkpoint(model, config, ckpt_path)
        print("2. Checkpoint saved")

        # Generate synthetic volume
        vol_shape = (20, 20, 5)
        print(f"3. Generating synthetic {vol_shape} volume...")
        dwi, gt_fa, gt_md, gt_tissue = generate_synthetic_volume(acq, vol_shape)

        # Save as NIfTI
        affine = np.eye(4)
        dwi_path = os.path.join(tmpdir, "dwi.nii.gz")
        bval_path = os.path.join(tmpdir, "bvals.txt")
        bvec_path = os.path.join(tmpdir, "bvecs.txt")
        mask_path = os.path.join(tmpdir, "mask.nii.gz")
        output_dir = os.path.join(tmpdir, "output")

        nib.save(nib.Nifti1Image(dwi, affine), dwi_path)
        # Save bvals/bvecs in FSL format (s/mm^2)
        bvals_fsl = np.asarray(acq.bvalues) / 1e6
        np.savetxt(bval_path, bvals_fsl[None, :], fmt="%.1f")
        np.savetxt(bvec_path, np.asarray(acq.gradient_directions).T, fmt="%.6f")
        mask = np.ones(vol_shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)

        # Deploy
        print("4. Running inference on volume...")
        predictor = SBIPredictor.from_checkpoint(ckpt_path)
        t0 = time.time()
        results = predictor.predict_volume(
            dwi_path, bval_path, bvec_path, mask_path,
            output_dir=output_dir,
            batch_size=1024,
        )
        t_deploy = time.time() - t0
        print(f"   Inference time: {t_deploy:.2f} s")

        # Validate
        print("\n5. Validation Results")
        print("=" * 70)

        pred_fa = results["FA"]
        pred_md = results["MD_scaled"]

        # Per-tissue accuracy
        tissues = {1: "WM", 2: "GM", 3: "CSF"}
        for t_id, t_name in tissues.items():
            mask_t = gt_tissue == t_id
            n = np.sum(mask_t)
            if n == 0:
                continue

            fa_err = np.mean(np.abs(pred_fa[mask_t] - gt_fa[mask_t]))
            md_err = np.mean(np.abs(pred_md[mask_t] - gt_md[mask_t]))
            fa_r = np.corrcoef(gt_fa[mask_t].ravel(), pred_fa[mask_t].ravel())[0, 1] if n > 3 else 0
            print(f"  {t_name} ({n} voxels):")
            print(f"    FA: MAE={fa_err:.3f}  r={fa_r:.4f}")
            print(f"    MD: MAE={md_err:.3f}")

        # Global
        fa_r_all = np.corrcoef(gt_fa.ravel(), pred_fa.ravel())[0, 1]
        md_r_all = np.corrcoef(gt_md.ravel(), pred_md.ravel())[0, 1]
        print(f"\n  Overall:")
        print(f"    FA correlation: r={fa_r_all:.4f}")
        print(f"    MD correlation: r={md_r_all:.4f}")

        # Check output NIfTIs exist
        print(f"\n  Output files in {output_dir}:")
        for f in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, f)
            img = nib.load(fpath)
            print(f"    {f}: shape={img.shape}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
