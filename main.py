#!/usr/bin/env python3
import json
import os
import sys
import jax

# Add the source directory to path if strictly necessary, 
# though pip install . in Dockerfile should handle it.
# sys.path.append('.') 

def main():
    # Load configuration
    with open('config.json') as f:
        config = json.load(f)

    # Validate inputs
    dwi_path = config.get('dwi')
    bvecs_path = config.get('bvecs')
    bvals_path = config.get('bvals')

    if not dwi_path or not os.path.exists(dwi_path):
        print(f"Error: DWI file not found at {dwi_path}")
        sys.exit(1)

    print(f"Starting dmipy-jax processing on {dwi_path}")
    
    # Check JAX devices
    print(f"JAX Devices: {jax.devices()}")

    # Placeholder for actual processing logic
    # In a real app, we would import dmipy_jax, load data, fit model, save output.
    # For now, we just demonstrate parameters were read and JAX is working.
    
    print("Processing complete (Placeholder).")

if __name__ == '__main__':
    main()
