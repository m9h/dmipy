
"""
CHAP Dataset Example: Real-World Data Showcase
---------------------------------------------
This script demonstrates fetching and analyzing three major public datasets:
1. Connectome 2.0 (ds006181): High-gradient, advanced microstructure.
2. EDDEN (ds004666): Denoising evaluation challenge data.
3. Synaesthesia (ds004466): Multi-shell diffusion data.

Usage:
    uv run python examples/demo_chap_real_data.py --dataset connectome2
    uv run python examples/demo_chap_real_data.py --dataset edden
    uv run python examples/demo_chap_real_data.py --dataset synaesthesia
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from dmipy_jax.data.public_datasets import (
    fetch_connectome2, fetch_edden, fetch_synaesthesia, get_dwi_path_generic
)
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.fitting.optimization import OptimistixFitter
from dipy.io.gradients import read_bvals_bvecs
from dmipy_jax.acquisition import JaxAcquisition

def main():
    parser = argparse.ArgumentParser(description="CHAP Real Data Demo")
    parser.add_argument("--dataset", type=str, choices=['connectome2', 'edden', 'synaesthesia'], 
                        default='connectome2', help="Dataset to analyze")
    args = parser.parse_args()

    print(f"--- Processing {args.dataset.upper()} Dataset ---")
    
    # 1. Fetch Data
    if args.dataset == 'connectome2':
        data_root = fetch_connectome2()
    elif args.dataset == 'edden':
        data_root = fetch_edden()
    else:
        data_root = fetch_synaesthesia()
        
    print(f"Dataset location: {data_root}")
    
    # 2. Find DWI file
    try:
        dwi_path = get_dwi_path_generic(data_root)
        print(f"DWI File: {dwi_path.name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Note: DataLad fetch might have only retrieved metadata. 'datalad get' is needed for content.")
        print(f"Try running: cd {data_root} && datalad get .")
        return

    # 3. Load Data (files might be annex pointers, check existence)
    if not dwi_path.exists() or dwi_path.stat().st_size < 1000:
        print("\n[!] File appears to be a DataLad pointer. Downloading content...")
        import subprocess
        subprocess.run(["datalad", "get", str(dwi_path)], check=True)
        
    img = nib.load(dwi_path)
    data = img.get_fdata()
    
    # Load bvals/bvecs (Naively assume they are next to the dwi)
    bval_path = dwi_path.with_suffix('').with_suffix('.bval')
    bvec_path = dwi_path.with_suffix('').with_suffix('.bvec')
    
    if not bval_path.exists():
        # Try .nii.gz -> .bval
        bval_path = Path(str(dwi_path).replace('.nii.gz', '.bval'))
        bvec_path = Path(str(dwi_path).replace('.nii.gz', '.bvec'))
    
    if not bval_path.exists():
         print("Could not find bvals/bvecs automatically. Skipping fit.")
         return
         
    # Ensure they are fetched too
    if not bval_path.exists() or bval_path.stat().st_size < 10:
         import subprocess
         subprocess.run(["datalad", "get", str(bval_path)], check=True)
         subprocess.run(["datalad", "get", str(bvec_path)], check=True)

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    
    # 4. Define Model
    ball = gaussian_models.Ball()
    stick = cylinder_models.Stick()
    zeppelin = gaussian_models.Zeppelin()
    
    model = JaxMultiCompartmentModel(models=[stick, zeppelin, ball])
    
    # 5. Fit a small slice
    print("Fitting model to center slice...")
    mid_x, mid_y, mid_z = np.array(data.shape[:3]) // 2
    roi_data = data[mid_x-5:mid_x+5, mid_y-5:mid_y+5, mid_z:mid_z+1, :]
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    fitter = OptimistixFitter(model, acq)
    fitted_params = fitter.fit(roi_data)
    
    print("Fit Complete!")
    print("Sample Fitted Parameters (Pixel 0,0):")
    for k, v in fitted_params.items():
        print(f"  {k}: {v[0,0,0]}")

if __name__ == "__main__":
    main()
