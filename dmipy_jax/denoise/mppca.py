
import jax
import jax.numpy as jnp
from functools import partial

def extract_patches(data, patch_radius):
    """
    Extract 3D patches from 4D Data (X, Y, Z, B) for MP-PCA.
    
    Args:
        data: (X, Y, Z, B) array.
        patch_radius: int or tuple (rx, ry, rz).
    
    Returns:
        patches: (N_voxels, M, B) where M is the number of voxels in a patch.
        spatial_shape: (X, Y, Z) tuple.
    """
    if isinstance(patch_radius, int):
        px, py, pz = (patch_radius,) * 3
    else:
        px, py, pz = patch_radius
        
    k_shape = (2*px+1, 2*py+1, 2*pz+1)
    
    # Data is (X, Y, Z, B).
    # Move B to channel dimension (and add batch=1) for conv_general_dilated_patches?
    # Actually, conv_general_dilated_patches operates on N, C, D, H, W.
    # Let's treat B as Channel C? 
    # If we treat B as C, we extract spatial patches for ALL B simultaneously.
    # Output of conv_patches with filter (kz, ky, kx) on input (N=1, C=B, D, H, W)
    # is (N=1, C=B * patch_vol, D_out, H_out, W_out).
    
    # Prepare input: (1, B, X, Y, Z)
    data_perm = jnp.transpose(data, (3, 0, 1, 2))[None, ...]
    
    patches = jax.lax.conv_general_dilated_patches(
        lhs=data_perm,
        filter_shape=k_shape,
        window_strides=(1, 1, 1),
        padding='SAME',
        dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
    )
    # Output shape: (1, B * M, X, Y, Z) where M = patch_volume
    
    # We want to reorganize this into (N_voxels, M, B).
    # Current: (1, B*M, X, Y, Z).
    # Reshape keys:
    # 1. (B, M, X, Y, Z)
    _, BM, X, Y, Z = patches.shape
    M = k_shape[0] * k_shape[1] * k_shape[2]
    B = BM // M
    
    patches = patches.reshape(B, M, X, Y, Z)
    
    # Transpose to (X, Y, Z, M, B)
    patches = jnp.transpose(patches, (2, 3, 4, 1, 0))
    
    # Flatten spatial: (N_voxels, M, B)
    patches_flat = patches.reshape(X*Y*Z, M, B)
    
    return patches_flat, (X, Y, Z)

@jax.jit
def mppca_kernel(patch_matrix, center_idx):
    """
    Core MP-PCA denoising kernel for a single patch matrix.
    Vectorized implementation to avoid control flow.
    
    Args:
        patch_matrix: (M, N) array where M is voxels in patch, N is measurements.
        center_idx: Index of the center voxel within the M voxels.
        
    Returns:
        denoised_signal: (N,) vector.
    """
    M, N = patch_matrix.shape
    
    # Centering
    mu = jnp.mean(patch_matrix, axis=0)
    Y = patch_matrix - mu
    
    # SVD
    # Y = U S V^T
    # Full matrices=False -> U:(M, K), S:(K,), Vt:(K, N) where K=min(M,N)
    U, s, Vt = jax.scipy.linalg.svd(Y, full_matrices=False)
    
    # Eigenvalues of covariance matrix
    # lambda = s^2 / (M - 1)
    vals = (s**2) / (M - 1)
    
    K = len(vals)
    arange_K = jnp.arange(K)
    
    # Vectorized Noise Estimation for all possible cutoffs p
    # sigma^2[p] is the mean of vals[p:]
    # Calculate suffix means using cumulative sums
    
    vals_rev = vals[::-1]
    cumsum_rev = jnp.cumsum(vals_rev)
    cumsum = cumsum_rev[::-1]
    
    # counts[p] is the number of elements in vals[p:] which is K - p
    counts = K - arange_K
    sigma2_all = cumsum / counts
    
    # Gamma calculation for all p
    # gamma[p] = (N - p) / (M - p)
    # Note: Use float division
    gamma_all = (N - arange_K) / (M - arange_K)
    
    # Threshold tau for all p
    tau_all = sigma2_all * (1.0 + jnp.sqrt(gamma_all))**2
    
    # Determine signal components
    # Condition: vals[p] > tau[p]
    is_signal = vals > tau_all
    
    # We want the first index p where condition fails (transition from signal to noise).
    # Assuming sorted eigenvalues (descending), signal are at the beginning.
    # So we look for the sequence [True, True, ..., True, False, ...]
    # The number of signal components P is the count of initial True values.
    # We can enforce monotonicity or just take the cumulative product.
    
    mask_monotonic = jnp.cumprod(is_signal)
    
    # Number of signal components
    # Cast to same type as arithmetic
    # Using sum implies we take all components that pass the check in the initial block.
    # However, for soft masking we could use mask_monotonic directly.
    
    # Reconstruct
    # signal = U[:, :p] @ S[:p, :p] @ Vt[:p, :]
    # U_c = U[center_idx, :] # (K,)
    # scale = U_c * S * mask
    
    U_c = U[center_idx, :]
    # Only keep components where mask_monotonic is 1
    scale = U_c * s * mask_monotonic
    
    recon = jnp.dot(scale, Vt)
    
    return recon + mu

def mppca(data, patch_radius=2):
    """
    MP-PCA Denoising function.
    
    Args:
        data: (X, Y, Z, B) dMRI data.
        patch_radius: Radius of the sliding window (default 2 -> 5x5x5 patches).
        
    Returns:
        denoised_data: (X, Y, Z, B)
    """
    X, Y, Z, B = data.shape
    
    # 1. Extract Patches
    # patches: (N_vox, M, B)
    patches_flat, spat_shape = extract_patches(data, patch_radius)
    
    # 2. Determine center index of the patch
    # The patch is flattened from (2r+1, 2r+1, 2r+1).
    # The center is at index M // 2.
    dim = 2 * patch_radius + 1
    M = dim ** 3
    center_idx = M // 2
    
    # 3. Vmap kernel over voxels
    denoise_fn = jax.vmap(partial(mppca_kernel, center_idx=center_idx))
    
    # Run
    denoised_flat = denoise_fn(patches_flat) # (N_vox, B)
    
    # 4. Reshape back
    return denoised_flat.reshape(X, Y, Z, B)

