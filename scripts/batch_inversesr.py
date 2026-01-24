#!/usr/bin/env python3
"""
Batch execution script for InverseSR on ds001957.
Patches InverseSR codebase to run on GPU and handle custom paths.
"""

import sys
import os
import glob
import logging
import argparse
from pathlib import Path
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inversesr_batch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATASET_ROOT = Path("/home/mhough/datasets/ds001957-study")
PREPROC_DIR = DATASET_ROOT / "derivatives" / "preproc_qsiprep"
OUTPUT_ROOT = DATASET_ROOT / "derivatives" / "inversesr"
INVERSESR_ROOT = Path("/home/mhough/dev/dmipy/benchmarks/external/InverseSR")

# Add InverseSR to path to allow imports
sys.path.insert(0, str(INVERSESR_ROOT))
sys.path.insert(0, str(INVERSESR_ROOT / "project"))

# --- Monkeypatching (Global) ---
# We must patch utils BEFORE importing BRGM_decoder because BRGM_decoder uses 'from utils.utils import ...'
import utils.const as const
import utils.utils as utils

# Global state for subject-specific paths
CURRENT_INPUT_PATH = None
CURRENT_OUTPUT_DIR = None

# 1. Patch Constants
const.PRETRAINED_MODEL_DECODER_PATH = INVERSESR_ROOT / "decoder"
const.PRETRAINED_MODEL_DDPM_PATH = INVERSESR_ROOT / "ddpm"
const.PRETRAINED_MODEL_VGG_PATH = INVERSESR_ROOT / "vgg16.pt"

# 2. Patch Utils
original_load_target_image = utils.load_target_image

def patched_load_target_image(hparams, device):
    global CURRENT_INPUT_PATH
    if CURRENT_INPUT_PATH is None:
        raise RuntimeError("CURRENT_INPUT_PATH not set")
    logger.info(f"Loading T1 from: {CURRENT_INPUT_PATH}")
    img_tensor = utils.transform_img(CURRENT_INPUT_PATH, device=device)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

utils.load_target_image = patched_load_target_image

def patched_load_ddpm_latent_vectors(device):
    path = INVERSESR_ROOT / "trained_models" / "latent_vector_ddpm_samples_100000.pt"
    if not path.exists():
         logger.error(f"Missing resource: {path}")
         raise FileNotFoundError(
             f"Please download 'latent_vector_ddpm_samples_100000.pt' to {path.parent}. "
             "This file is required by InverseSR."
         )
    return torch.load(path, map_location=device)

def patched_load_latent_vector_stats(device):
    path = INVERSESR_ROOT / "trained_models" / "latent_vector_mean_std.pt"
    if not path.exists():
         logger.error(f"Missing resource: {path}")
         raise FileNotFoundError(
             f"Please download 'latent_vector_mean_std.pt' to {path.parent}. "
             "This file is required by InverseSR."
         )
    tmp = torch.load(path, map_location=device)
    return tmp["latent_vector_mean"], tmp["latent_vector_std"]

utils.load_ddpm_latent_vectors = patched_load_ddpm_latent_vectors
utils.load_ddpm_latent_vectors = patched_load_ddpm_latent_vectors
utils.load_latent_vector_stats = patched_load_latent_vector_stats

# 3. Patch setup_noise_inputs (Upstream Bug Fix)
# BRGM_decoder.py calls this without hparams, but utils.py requires it.
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


# 4. Patch VGG Perceptual (Upstream Missing File)
# Check for VGG presence and fallback GLOBALLY
if not const.PRETRAINED_MODEL_VGG_PATH.exists():
    logger.info(f"VGG16 model missing at {const.PRETRAINED_MODEL_VGG_PATH}. Using torchvision fallback.")
    
    try:
            from torchvision.models import vgg16, VGG16_Weights
    except ImportError:
            logger.error("torchvision not installed. Cannot use fallback. Please install torchvision.")
            raise

    class VGGPerceptualWrapper(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            logger.info("Downloading VGG16 weights from torchvision...")
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
            # Extract features up to relu3_3 (index 16 in features usually)
            self.features = vgg.features[:16].eval()
            for param in self.features.parameters():
                param.requires_grad = False
        
        def forward(self, x, resize_images=False, return_lpips=True):
                return self.features(x)
                
    def patched_load_vgg_perceptual(hparams, target, device):
        vgg = VGGPerceptualWrapper(device)
        # We also need to compute target features
        target_features = utils.getVggFeatures(hparams, target, vgg)
        return vgg, target_features
        
    utils.load_vgg_perceptual = patched_load_vgg_perceptual


# --- Import Project Logic (AFTER Patching) ---
from project.BRGM_decoder import project as run_project
from project.BRGM_decoder import load_pre_trained_decoder
from project.utils.utils import create_corruption_function

def run_subject(sub_id, input_t1, output_dir, device_str="cuda"):
    """
    Runs InverseSR for a single subject.
    """
    global CURRENT_INPUT_PATH, CURRENT_OUTPUT_DIR
    logger.info(f"Processing Subject: {sub_id}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Update global state for patches
    CURRENT_INPUT_PATH = input_t1
    CURRENT_OUTPUT_DIR = output_dir
    const.INPUT_FOLDER = input_t1.parent
    const.OUTPUT_FOLDER = output_dir

    # CRITICAL FIX: BRGM_decoder imports OUTPUT_FOLDER using 'from ... import', so it has a copy.
    # We must patch the copy inside the module itself.
    import project.BRGM_decoder
    project.BRGM_decoder.OUTPUT_FOLDER = output_dir

    # Arguments (mimicking argparse Namespace)
    hparams = argparse.Namespace(
        subject_id=sub_id,
        experiment_name=f"{sub_id}_inversesr",
        tensor_board_logger=str(output_dir / "logs"),
        data_format="nii",
        corruption="None",
        downsample_factor=1,
        mask_id="0",
        num_steps=601,
        learning_rate=7e-2,
        lambda_perc=1e4,
        prior_every=1,
        prior_after=45,
        n_samples=3,
        k=1,
        start_steps=0,
        update_latent_variables=True,
        update_conditioning=False,
        update_gender=False,
        update_age=False,
        update_ventricular=False,
        update_brain=False,
        alpha_downsampling_loss=0, 
        mean_latent_vector=False,
        perc_dim="axial",
        ddim_num_timesteps=250, 
        ddim_eta=0.0
    )

    # Arguments (mimicking argparse Namespace)
    hparams = argparse.Namespace(
        subject_id=sub_id,
        experiment_name=f"{sub_id}_inversesr",
        tensor_board_logger=str(output_dir / "logs"),
        data_format="nii",
        corruption="None",
        downsample_factor=1,
        mask_id="0",
        num_steps=601,
        learning_rate=7e-2,
        lambda_perc=1e4,
        prior_every=1,
        prior_after=45,
        n_samples=3,
        k=1,
        start_steps=0,
        update_latent_variables=True,
        update_conditioning=False,
        update_gender=False,
        update_age=False,
        update_ventricular=False,
        update_brain=False,
        alpha_downsampling_loss=0, 
        mean_latent_vector=False,
        perc_dim="axial",
        ddim_num_timesteps=250, 
        ddim_eta=0.0
    )

    # Arguments (mimicking argparse Namespace)
    hparams = argparse.Namespace(
        subject_id=sub_id,
        experiment_name=f"{sub_id}_inversesr",
        tensor_board_logger=str(output_dir / "logs"),
        data_format="nii",
        corruption="None",
        downsample_factor=1,
        mask_id="0",
        num_steps=601,
        learning_rate=7e-2,
        lambda_perc=1e4,
        prior_every=1,
        prior_after=45,
        n_samples=3,
        k=1,
        start_steps=0,
        update_latent_variables=True,
        update_conditioning=False,
        update_gender=False,
        update_age=False,
        update_ventricular=False,
        update_brain=False,
        alpha_downsampling_loss=0, 
        mean_latent_vector=False,
        perc_dim="axial",
        ddim_num_timesteps=250, 
        ddim_eta=0.0
    )

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load Data
        target_img = utils.load_target_image(hparams, device)
        
        # Load Model
        decoder = load_pre_trained_decoder(
            vae_path=const.PRETRAINED_MODEL_DECODER_PATH,
            device=device
        )
        
        # Create Corruption/Forward Model
        forward = create_corruption_function(hparams=hparams, device=device)
        
        # Tensorboard writer
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=hparams.tensor_board_logger)
        
        # Run Project (Optimization)
        run_project(
            vqvae=decoder,
            forward=forward,
            target=target_img,
            device=device,
            writer=writer,
            hparams=hparams,
            verbose=True
        )
        
        # Clean up
        writer.close()
        logger.info(f"Successfully processed {sub_id}")
        
    except Exception as e:
        logger.error(f"Failed processing {sub_id}: {e}", exc_info=True)
        # Check for model existence
        if not const.PRETRAINED_MODEL_DECODER_PATH.exists():
            logger.critical("Pretrained models missing! Please download 'decoder' and 'ddpm' to InverseSR folder.")
            return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print subjects and exit")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    # Find Subjects
    search_pattern = str(PREPROC_DIR / "sub-*" / "anat" / "t1.nii.gz")
    t1_files = glob.glob(search_pattern)
    t1_files.sort()
    
    if not t1_files:
        logger.error(f"No T1 files found in {PREPROC_DIR}")
        return

    logger.info(f"Found {len(t1_files)} subjects.")
    
    for t1_path in t1_files:
        t1_path = Path(t1_path)
        sub_id = t1_path.parent.parent.name # sub-XX
        
        output_dir = OUTPUT_ROOT / sub_id
        
        if args.dry_run:
            print(f"[Dry Run] would process {sub_id}")
            print(f"  Input: {t1_path}")
            print(f"  Output: {output_dir}")
            continue
            
        run_subject(sub_id, t1_path, output_dir, device_str=args.device)

if __name__ == "__main__":
    main()
