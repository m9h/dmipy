import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
import os
import sys
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme
# Import STE from sibling file (needs path hack if package not installed, but usually okay in scripts)
# Or define a dummy one if import fails.

def compute_fd_gradient(volume):
    """
    Compute finite difference gradient magnitude of a 3D volume.
    Args:
        volume: (X, Y, Z) float array.
    Returns:
        grad_mag: (X, Y, Z) gradient magnitude.
    """
    # jnp.gradient returns a list of arrays [d/dx, d/dy, d/dz]
    grads = jnp.gradient(volume)
    # Stack them: (3, X, Y, Z)
    grad_stack = jnp.stack(grads, axis=0)
    # Compute magnitude
    grad_mag = jnp.linalg.norm(grad_stack, axis=0)
    return grad_mag, grad_stack

def create_synthetic_phantom(shape=(64, 64, 64), radius=20):
    """Creates a sphere phantom."""
    x = jnp.linspace(-32, 32, shape[0])
    y = jnp.linspace(-32, 32, shape[1])
    z = jnp.linspace(-32, 32, shape[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    R = jnp.sqrt(X**2 + Y**2 + Z**2)
    # Signal is 1 inside, 0 outside
    phantom = jnp.where(R < radius, 1.0, 0.0)
    # Smooth it a bit to make gradients nicer?
    # No, leave it binary to test edge detection, or use sigmoid for soft edge.
    phantom = 1.0 / (1.0 + jnp.exp(R - radius))
    return phantom

def main():
    print("Running Spatial Gradient Validator...")
    
    # Try to load data
    # Try to load data from manifest
    manifest_path = "experiments/ste_dataset_integration/data_manifest.json"
    import json
    data_path = None
    
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            # Prefer ex-vivo for validation as it is cleaner
            data_path = manifest.get('ex_vivo')
            print(f"Loaded path from manifest: {data_path}")
            
    output_path = "experiments/ste_dataset_integration/grad_magnitude.nii.gz"
    
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        img = nib.load(data_path)
        data = jnp.array(img.get_fdata())
        # Take first volume if 4D
        if data.ndim == 4:
            vol = data[..., 0]
        else:
            vol = data
        affine = img.affine
    else:
        print(f"Data not found at {data_path}. using SYNTHETIC PHANTOM.")
        vol = create_synthetic_phantom()
        affine = np.eye(4)
        
    print(f"Volume shape: {vol.shape}")
    
    # Compute Gradient
    grad_mag, grad_stack = compute_fd_gradient(vol)
    
    print(f"Gradient Magnitude computed. Mean: {jnp.mean(grad_mag):.4f}, Max: {jnp.max(grad_mag):.4f}")
    
    # Save result
    # Convert back to numpy for nibabel
    grad_mag_np = np.array(grad_mag)
    out_img = nib.Nifti1Image(grad_mag_np, affine)
    nib.save(out_img, output_path)
    print(f"Saved gradient magnitude to {output_path}")

if __name__ == "__main__":
    main()
