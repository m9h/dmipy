import numpy as np
import nibabel as nib
import os
import subprocess
import sys

def verify_swinir():
    # 1. Check dependencies
    try:
        import timm
        print("timm is installed.")
    except ImportError:
        print("Error: timm is not installed. SwinIR requires timm.")
        print("Please install it running: pip install timm") # We can't run pip directly safely without user knowing
        # Check if we can proceed or fail
        # return
    
    # 2. Generate dummy data
    input_file = "test_input.nii.gz"
    output_file = "test_output.nii.gz"
    
    # Shape: 32x32 spatial, 2 slices, 2 directions
    # Small enough to be fast
    data = np.random.rand(32, 32, 2, 2).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, input_file)
    print(f"Created dummy input: {input_file} {data.shape}")
    
    # 3. Run benchmark script
    # We call the python script we just made
    cmd = [sys.executable, "benchmarks/run_swinir.py", input_file, output_file, "--scale", "4", "--device", "cpu"] 
    # Force CPU for verification to avoid CUDA OOM or issues if user has no GPU
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print("Execution failed.")
        return
    
    # 4. Check output
    if not os.path.exists(output_file):
        print("Output file was not created.")
        return
        
    out_img = nib.load(output_file)
    out_data = out_img.get_fdata()
    print(f"Output shape: {out_data.shape}")
    
    expected_shape = (32*4, 32*4, 2, 2)
    if out_data.shape == expected_shape:
        print("SUCCESS: Output shape matches expected upscaling.")
    else:
        print(f"FAILURE: Output shape {out_data.shape} != expected {expected_shape}")
        
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

if __name__ == "__main__":
    verify_swinir()
