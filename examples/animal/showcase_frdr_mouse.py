import os
import argparse
import nibabel as nib
import numpy as np

from dmipy_jax.io.frdr import FRDRMouseLoader
from dmipy_jax.pipelines.rodent import RodentPreprocessing

def main():
    parser = argparse.ArgumentParser(description="Showcase FRDR Mouse Pipeline")
    parser.add_argument("--frdr-root", type=str, required=True, help="Path to FRDR dataset root")
    parser.add_argument("--subject", type=str, default="01", help="Subject ID")
    parser.add_argument("--session", type=str, default="01", help="Session ID")
    parser.add_argument("--output-dir", type=str, default="./frdr_output", help="Output directory")
    args = parser.parse_args()
    
    # 1. Initialize Loader
    print(f"Initializing FRDR Loader at {args.frdr_root}")
    loader = FRDRMouseLoader(args.frdr_root)
    
    try:
        data = loader.load_subject(args.subject, args.session)
    except FileNotFoundError as e:
        print(f"Error loading subject: {e}")
        return

    print(f"Found Data:")
    print(f"  DWI: {data['dwi_nii']}")
    print(f"  Anat: {data['anat_nii']}")
    
    # 2. Pipeline Setup
    print(f"Initializing Rodent Pipeline -> {args.output_dir}")
    pipeline = RodentPreprocessing(args.output_dir)
    
    # 3. Preprocessing Steps
    
    # A. N4 on Anatomical
    # Check if N4 is already done or run it
    t2w_n4 = pipeline.run_n4_bias_correction(data['anat_nii'])
    print(f"  [N4] Corrected T2w saved to {t2w_n4}")
    
    # B. Compute Mean b0 from DWI for Registration
    # Load DWI, extract b0s, mean
    print("  [Reg] Computing Mean b0...")
    img = nib.load(data['dwi_nii'])
    dwi_data = img.get_fdata()
    bvals = np.loadtxt(data['bval'])
    
    # Simple b0 mask (b < 50)
    b0_mask = bvals < 50
    if np.sum(b0_mask) == 0:
        print("Error: No b0 volumes found!")
        return
        
    mean_b0 = np.mean(dwi_data[..., b0_mask], axis=-1)
    
    # Save Mean b0 for registration
    mean_b0_path = os.path.join(args.output_dir, f"sub-{args.subject}_sess-{args.session}_meanb0.nii.gz")
    nib.save(nib.Nifti1Image(mean_b0, img.affine), mean_b0_path)
    print(f"  [Reg] Mean b0 saved to {mean_b0_path}")
    
    # C. Register Mean b0 -> T2w_N4
    # Note: Usually we register DWI to T2w (Anat space) or T2w to DWI
    # The Knowledge Item suggests: "Align mean b=0 image (DWI space) to corrected T2w (Anatomical space)"
    tx_path = pipeline.register_dwi_to_anat(mean_b0_path, t2w_n4)
    print(f"  [Reg] Registration complete. Transform: {tx_path}")
    
    # D. Verification - Resample Mean b0 to Anat space
    resampled_b0 = pipeline.apply_transform(mean_b0_path, t2w_n4, tx_path)
    print(f"  [Verify] Resampled Mean b0 saved to {resampled_b0}")
    
    print("\nPipeline Showcase Complete.")
    print("Next Steps: Visualize T2w_N4 and Resampled_Mean_b0 in FSL/ITK-SNAP to verify alignment.")

if __name__ == "__main__":
    main()
