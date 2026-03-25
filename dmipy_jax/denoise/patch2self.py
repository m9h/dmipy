
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

def extract_3d_patches(data, patch_radius):
    """
    Extract 3D patches from 4D Data (X, Y, Z, B).
    
    Args:
        data: (X, Y, Z, B) array.
        patch_radius: int or tuple (rx, ry, rz).
    
    Returns:
        patches: (N_voxels, B, Patch_Size)
        spatial_shape: (X, Y, Z)
    """
    if isinstance(patch_radius, int):
        px, py, pz = (patch_radius,) * 3
    else:
        px, py, pz = patch_radius
        
    k_shape = (2*px+1, 2*py+1, 2*pz+1)
    
    # 1. Permute to (B, Z, Y, X) for conv_general_dilated_patches
    # JAX expects (N, C, Spatial dims). 
    # We treat 'B' as Batch 'N' for extraction, and 1 as 'C'? 
    # Or just use (1, B, Z, Y, X) and extract over channel B?
    # No, we want spatial patches. 'B' should be preserved.
    # Let's treat B as 'Depth' or 'Channel'?
    # It's easier to vmap extraction over B, or treat B as Batch.
    
    B_dim = data.shape[-1]
    
    # (B, X, Y, Z)
    data_b = jnp.transpose(data, (3, 0, 1, 2))
    
    # Add channel dim for conv: (B, 1, X, Y, Z)
    data_b = data_b[:, None, :, :, :]
    
    # Patch extraction
    # Output: (B, 1 * kx*ky*kz, out_X, out_Y, out_Z)
    # We want padding='SAME' to keep output size same as input
    
    patches = jax.lax.conv_general_dilated_patches(
        lhs=data_b,
        filter_shape=k_shape,
        window_strides=(1, 1, 1),
        padding='SAME',
        dimension_numbers=('NCDHW', 'OIHW', 'NCDHW') # 3D convention
        # Check specific dimension numbers for 5D input (N, C, D, H, W)
        # Standard is 'NCDHW' ?
        # Actually standard conv_general_dilated_patches usually takes 
        # dimension_numbers argument.
        # For 3D: N, C, D, H, W.
    )
    
    # Shape of patches: (B, PatchSize, X, Y, Z)
    
    # Transpose to (X, Y, Z, B, PatchSize)
    # patches = jnp.transpose(patches, (2, 3, 4, 0, 1))
    
    # Flatten spatial: (N_voxels, B, PatchSize)
    X, Y, Z = data.shape[:3]
    return patches.reshape(B_dim, -1, X*Y*Z).transpose(2, 0, 1)

@partial(jax.jit, static_argnames=['patch_radius', 'model'])
def patch2self_kernel(data, patch_radius=1, model='ols', alpha=1.0):
    """
    JAX Kernel for Patch2Self.
    
    Args:
        data: (X, Y, Z, B)
        patch_radius: int
        model: 'ols' or 'ridge'
        alpha: regularization for ridge
    
    Returns:
        denoised_data: (X, Y, Z, B)
    """
    X_dim, Y_dim, Z_dim, B_dim = data.shape
    
    # 1. Extract Patches: (N_voxels, B, P_size)
    # We need to act carefully with memory.
    # If 100^3 voxels, 60 vols, 27 patch -> 1.6e9 floats -> 6.4 GB.
    # This might fit.
    
    # Helper to get patches
    px, py, pz = (patch_radius, patch_radius, patch_radius) if isinstance(patch_radius, int) else patch_radius
    k_shape = (2*px+1, 2*py+1, 2*pz+1)
    
    # (B, X, Y, Z)
    data_b = jnp.transpose(data, (3, 0, 1, 2))
    # (B, 1, X, Y, Z)
    data_b = data_b[:, None, :, :, :]
    
    patches_b = jax.lax.conv_general_dilated_patches(
        lhs=data_b,
        filter_shape=k_shape,
        window_strides=(1, 1, 1),
        padding='SAME',
        dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
    )
    # (B, P_size, X, Y, Z)
    
    # Flatten spatial dims to facilitate regression
    # (B, P_size, N_vox)
    patches_flat = patches_b.reshape(B_dim, -1, X_dim*Y_dim*Z_dim)
    # Transpose to (N_vox, B, P_size) for easier indexing? 
    # Or keep (B, ...) to slice volumes easily?
    
    # Let's keep (B, P_size, N_vox) so we can construct Feature Matrix X easily.
    # Wait, features are (N_vox, Features).
    # X_feat = All volumes except 'i'.
    # X_feat: (N_vox, (B-1)*P_size)
    
    # It is better to have (N_vox, B, P_size).
    patches_flat = jnp.transpose(patches_flat, (2, 0, 1)) # (N_vox, B, P_size)
    
    # 2. Main Loop over volumes
    def process_volume(i):
        # Target: volume i
        # y: (N_vox, 1)
        y = data[..., i].reshape(-1, 1)
        
        # Features: All patches except volume i
        # Roll patches so the target volume i is at the end?
        # No, roll(-i) brings index i to position 0.
        p_rolled = jnp.roll(patches_flat, -i, axis=1)
        
        # We want to exclude index i (now at 0).
        X_feat = p_rolled[:, 1:, :] # (N_vox, B-1, P_size)
        
        # Flatten feature dimension
        X_feat = X_feat.reshape(X_feat.shape[0], -1) # (N_vox, (B-1)*P_size)
        
        # Add Bias (Intercept)
        bias = jnp.ones((X_feat.shape[0], 1))
        X_feat = jnp.hstack([X_feat, bias])
        
        # OLS: beta = (X'X)^-1 X'y
        # Compute Gram Matrix
        XTX = jnp.dot(X_feat.T, X_feat)
        XTy = jnp.dot(X_feat.T, y)
        
        if model == 'ridge':
             XTX = XTX + alpha * jnp.eye(XTX.shape[0])
        elif model == 'ols':
             # Add tiny regularization for numerical stability
             XTX = XTX + 1e-6 * jnp.eye(XTX.shape[0])

        # Solve
        beta = jax.scipy.linalg.solve(XTX, XTy, assume_a='pos')
        
        # Predict
        y_pred = jnp.dot(X_feat, beta)
        
        return y_pred.reshape(X_dim, Y_dim, Z_dim)

    # Scan over all volumes sequentially to save memory
    output = jax.lax.map(process_volume, jnp.arange(B_dim))
    
    # output is (B, X, Y, Z). Transpose back to (X, Y, Z, B)
    return jnp.transpose(output, (1, 2, 3, 0))

class Patch2Self:
    def __init__(self, patch_radius=1, model='ols'):
        self.patch_radius = patch_radius
        self.model = model
        
    def __call__(self, data, bvals=None):
        return patch2self_kernel(data, self.patch_radius, self.model)
