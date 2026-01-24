import argparse
import amico
import os
import glob
import shutil

def run_amico_fit(data_path, bvals_path, bvecs_path, mask_path, scheme_file, model_name, output_dir, **kwargs):
    """
    Internal function to run AMICO fit inside the container.
    """
    # 1. Setup AMICO
    amico.core.setup()

    # 2. Load Data
    # AMICO requires data to be in a specific subject folder structure or loaded explicitly.
    # We will use the explicit loading if possible, or organize the temp dir to match.
    # AMICO usually expects: /path/to/subject/ and then loads 'DWI.nii.gz', etc.
    
    # Let's assume the data is mounted at /data and files are named specifically
    # specific filenames are often expected by AMICO.load_data, or we can use 
    # the underlying module functions directly. 
    
    # However, standard AMICO workflow:
    # subject_folder = '/data'
    # amico.core.setup()
    # amico.util.fsl2scheme(bvals_path, bvecs_path, scheme_file)
    # amico.core.load_data(dwi_filename=data_path, scheme_filename=scheme_file, mask_filename=mask_path, b0_thr=0)
    
    # Generate scheme file from bvals/bvecs if it doesn't exist
    if not os.path.exists(scheme_file):
        print(f"Generating scheme file from {bvals_path} and {bvecs_path}")
        amico.util.fsl2scheme(bvals_path, bvecs_path, scheme_file)

    # 3. Load Data
    print("Loading data...")
    # subject_path is just the directory containing the files
    subject_path = os.path.dirname(data_path)
    
    # Note: load_data arguments might vary slightly by version, checking source would be ideal but 
    # we stick to standard API.
    amico.core.load_data(
        dwi_filename=os.path.basename(data_path),
        scheme_filename=os.path.basename(scheme_file),
        mask_filename=os.path.basename(mask_path),
        b0_thr=0,
        subject=subject_path
    )

    # 4. Set Model
    print(f"Setting model to {model_name}...")
    if model_name.upper() == 'NODDI':
        amico.core.set_model("NODDI")
        # Generate kernels - critical step
        amico.core.generate_kernels()
        # Load kernels
        amico.core.load_kernels()
    elif model_name.upper() == 'ACTIVEAX':
        # ActiveAX config might be more complex, basic setup for now
        amico.core.set_model("ActiveAx")
        amico.core.generate_kernels()
        amico.core.load_kernels()
    elif model_name.upper() == 'SANDI':
         amico.core.set_model("SANDI")
         amico.core.generate_kernels()
         amico.core.load_kernels()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 5. Fit
    print("Fitting model...")
    amico.core.fit()

    # 6. Save Results
    print(f"Saving results to {output_dir}...")
    amico.core.save_results(path_suffix=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMICO fit inside container")
    parser.add_argument("--data", required=True, help="Path to DWI NIfTI file")
    parser.add_argument("--bvals", required=True, help="Path to bvals file")
    parser.add_argument("--bvecs", required=True, help="Path to bvecs file")
    parser.add_argument("--mask", required=True, help="Path to mask NIfTI file")
    parser.add_argument("--scheme", default="scheme.txt", help="Output scheme filename")
    parser.add_argument("--model", required=True, help="Model name (NODDI, ActiveAx, SANDI)")
    parser.add_argument("--output", default="AMICO/RESULTS", help="Output directory suffix")

    args = parser.parse_args()

    run_amico_fit(
        args.data,
        args.bvals,
        args.bvecs,
        args.mask,
        args.scheme,
        args.model,
        args.output
    )
