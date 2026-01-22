#!/usr/bin/env python3
"""
Generates latent statistics (mean, std, samples) required by InverseSR.
Uses the pre-trained DDPM model to sample latent vectors.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INVERSESR_ROOT = Path("/home/mhough/dev/dmipy/benchmarks/external/InverseSR")
sys.path.insert(0, str(INVERSESR_ROOT))
sys.path.insert(0, str(INVERSESR_ROOT / "project"))

# Patch consts to direct output
import project.utils.const as const
const.OUTPUT_FOLDER = INVERSESR_ROOT / "trained_models"

# Ensure output dir exists
os.makedirs(const.OUTPUT_FOLDER, exist_ok=True)

from project.utils.utils import compute_prior_stats
import project.utils.utils as utils

# --- Monkeypatching ---
# The InverseSR code contains a bug where compute_prior_stats calls setup_noise_inputs without hparams.
# We patch it to handle this.

original_setup_noise_inputs = utils.setup_noise_inputs

def patched_setup_noise_inputs(device, hparams=None):
    if hparams is None:
        # Default conditioning values used in setup_noise_inputs:
        # gender = 0.5 (False), age=63 (False), ventricular=0.5 (False), brain=0.5 (False)
        # We construct a dummy Namespace
        hparams = argparse.Namespace(
            update_gender=False,
            update_age=False,
            update_ventricular=False,
            update_brain=False
        )
    return original_setup_noise_inputs(device, hparams)

utils.setup_noise_inputs = patched_setup_noise_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    ddpm_path = INVERSESR_ROOT / "ddpm"
    if not ddpm_path.exists():
        logger.error(f"DDPM model not found at {ddpm_path}. Please download it first.")
        sys.exit(1)

    logger.info(f"Generating {args.n_samples} samples from DDPM at {ddpm_path}...")
    logger.info("This may take a while...")

    mean, std = compute_prior_stats(
        diffusion_path=ddpm_path,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device
    )
    
    logger.info("Generation complete.")
    logger.info(f"Saved stats to {const.OUTPUT_FOLDER}")
    
    # Rename samples file to match what batch script might expect if it strictly looks for 100000
    # The utils.py saves as 'latent_vector_samples.pt'.
    # But utils.py *loads* 'latent_vector_ddpm_samples_100000.pt'.
    # We should rename or symlink.
    
    generated_samples = const.OUTPUT_FOLDER / "latent_vector_samples.pt"
    target_samples = const.OUTPUT_FOLDER / "latent_vector_ddpm_samples_100000.pt"
    
    if generated_samples.exists():
        logger.info(f"Renaming {generated_samples.name} to {target_samples.name}")
        os.rename(generated_samples, target_samples)
    else:
        logger.warning(f"Expected generated file {generated_samples} not found.")

if __name__ == "__main__":
    main()
