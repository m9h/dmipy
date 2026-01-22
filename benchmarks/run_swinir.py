import argparse
import nibabel as nib
import numpy as np
import os
import sys

# Ensure we can import the adapter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adapters.swinir_adapter import SwinIRPredictor

def main():
    parser = argparse.ArgumentParser(description="Run SwinIR Benchmark on 4D dMRI data")
    parser.add_argument("input_nii", type=str, help="Path to input dMRI NIfTI file")
    parser.add_argument("output_nii", type=str, help="Path to save super-resolved NIfTI file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to SwinIR .pth checkpoint")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_nii):
        print(f"Error: Input file {args.input_nii} does not exist.")
        return
        
    print(f"Loading {args.input_nii}...")
    img = nib.load(args.input_nii)
    data = img.get_fdata()
    affine = img.affine
    
    print(f"Input shape: {data.shape}")
    
    # Initialize Predictor
    predictor = SwinIRPredictor(model_path=args.model_path, scale=args.scale, device=args.device)
    
    # Run Prediction
    print("Starting inference...")
    sr_data = predictor.predict_volume(data)
    
    print(f"Output shape: {sr_data.shape}")
    
    # Update affine for new resolution
    # Assumes isotropic scaling on X and Y, and no change on Z (since we process slice-by-slice)
    # Actually, if we upscale in-plane X/Y, we must adjust the affine diagonal 0 and 1.
    new_affine = affine.copy()
    new_affine[0, 0] = affine[0, 0] / args.scale
    new_affine[1, 1] = affine[1, 1] / args.scale
    # new_affine[0, 3] and [1, 3] (offsets) might need adjustment depending on center/corner alignment
    # For now, simplistic scaling of voxel size.
    
    # Save Output
    print(f"Saving to {args.output_nii}...")
    sr_img = nib.Nifti1Image(sr_data, new_affine)
    nib.save(sr_img, args.output_nii)
    print("Done.")

if __name__ == "__main__":
    main()
