#!/usr/bin/env python3
"""
Comparison runner: DIPY conventional vs dmipy-JAX on DIPY multi_tensor phantom.

Generates a 3-zone synthetic phantom (WM / crossing / CSF) using DIPY's
multi_tensor simulator, then runs:
  - DIPY DTI (conventional least-squares)
  - DIPY DKI (if multi-shell)
  - dmipy-JAX SBI (from checkpoint, if provided)
  - dmipy-JAX dictionary matching (from library, if provided)

Produces comparison tables, scatter plots, and NIfTI outputs.

Usage::

    # DIPY-only comparison (no checkpoint needed):
    python validation/validate_comparison.py

    # With SBI checkpoint:
    python validation/validate_comparison.py --sbi-checkpoint checkpoints/dti_mdn

    # With dictionary library:
    python validation/validate_comparison.py --library checkpoints/dti_library.h5

    # Full comparison:
    python validation/validate_comparison.py \\
        --sbi-checkpoint checkpoints/dti_mdn \\
        --library checkpoints/dti_library.h5 \\
        --output-dir results/comparison
"""

import argparse
import sys

import numpy as np

from dmipy_jax.pipeline.comparison import (
    ComparisonRunner,
    generate_dipy_multitensor_phantom,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-method comparison on synthetic phantom"
    )
    parser.add_argument(
        "--sbi-checkpoint", type=str, default=None,
        help="Path to dmipy-JAX SBI checkpoint (without .eqx extension)",
    )
    parser.add_argument(
        "--library", type=str, default=None,
        help="Path to HDF5 simulation library for dictionary matching",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/comparison",
        help="Output directory for NIfTIs and plots",
    )
    parser.add_argument(
        "--vol-shape", type=int, nargs=3, default=[20, 20, 5],
        help="Phantom volume shape (default: 20 20 5)",
    )
    parser.add_argument(
        "--snr", type=float, default=30.0,
        help="Signal-to-noise ratio (default: 30)",
    )
    parser.add_argument(
        "--multi-shell", action="store_true",
        help="Generate multi-shell phantom (enables DKI comparison)",
    )
    args = parser.parse_args()

    vol_shape = tuple(args.vol_shape)

    # Generate acquisition
    if args.multi_shell:
        rng = np.random.RandomState(42)
        n_dirs = 30
        bvecs_gen = rng.randn(n_dirs, 3)
        bvecs_gen = bvecs_gen / np.linalg.norm(bvecs_gen, axis=1, keepdims=True)
        bvals = np.concatenate([
            np.zeros(2),
            np.full(n_dirs, 1000),
            np.full(n_dirs, 2000),
        ])
        bvecs = np.concatenate([
            np.array([[1, 0, 0], [0, 1, 0]]),
            bvecs_gen,
            bvecs_gen,  # same directions, different b-value
        ])
    else:
        bvals, bvecs = None, None  # use defaults

    print("=" * 70)
    print("Multi-Method Comparison on DIPY Multi-Tensor Phantom")
    print("=" * 70)

    print(f"\nGenerating {vol_shape} phantom (SNR={args.snr})...")
    data, bvals_fsl, bvecs, mask, gt = generate_dipy_multitensor_phantom(
        vol_shape=vol_shape, bvals_fsl=bvals, bvecs=bvecs, snr=args.snr,
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Measurements: {data.shape[-1]}")
    print(f"  GT FA range: [{gt['FA'].min():.3f}, {gt['FA'].max():.3f}]")

    # Build comparison runner
    runner = ComparisonRunner()
    runner.add_dipy_dti()

    if args.multi_shell:
        runner.add_dipy_dki()

    if args.sbi_checkpoint:
        runner.add_sbi(args.sbi_checkpoint, name="dmipy SBI")

    if args.library:
        runner.add_dictionary(args.library, name="dmipy Dict")

    # Run
    result = runner.run(
        data, bvals_fsl, bvecs,
        mask=mask, ground_truth=gt, bvals_unit="fsl",
    )

    # Report
    result.print_summary()

    # Save outputs
    print(f"\nSaving outputs to {args.output_dir}/")
    result.save_niftis(args.output_dir)

    # Scatter plots
    method_names = list(result.methods.keys())
    if len(method_names) >= 2:
        for i, ma in enumerate(method_names):
            for mb in method_names[i+1:]:
                plot_path = f"{args.output_dir}/{ma}_vs_{mb}.png"
                result.plot_scatter(ma, mb, save_path=plot_path)
                print(f"  Scatter plot: {plot_path}")

    # GT scatter plots
    if gt:
        for method in method_names:
            # Create a "GT" pseudo-method for scatter plotting
            result.add_method("Ground Truth", gt)
            common = set(result.methods[method].keys()) & set(gt.keys())
            if common:
                plot_path = f"{args.output_dir}/{method}_vs_GT.png"
                result.plot_scatter(method, "Ground Truth",
                                    metrics=sorted(common), save_path=plot_path)
                print(f"  GT scatter: {plot_path}")
            del result.methods["Ground Truth"]

    print("\nDone.")


if __name__ == "__main__":
    main()
