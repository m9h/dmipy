import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Tuple, Any, Optional
import functools

# --- Helpers ---

def get_tensor_orientation(D_flat):
    """
    Decomposes flat tensor (6,) into principal eigenvector.
    D_flat: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    """
    # Construct 3x3 matrix
    # [ Dxx Dxy Dxz ]
    # [ Dxy Dyy Dyz ]
    # [ Dxz Dyz Dzz ]
    
    D_mat = jnp.array([
        [D_flat[0], D_flat[1], D_flat[2]],
        [D_flat[1], D_flat[3], D_flat[4]],
        [D_flat[2], D_flat[4], D_flat[5]]
    ])
    
    # Eigendecomposition
    # jnp.linalg.eigh for symmetric
    evals, evecs = jnp.linalg.eigh(D_mat)
    
    # eigh returns ascending order usually. Principal is last.
    e1 = evecs[:, 2] # (3,)
    return e1

# --- Probabilistic Tracker ---

class ProbabilisticTracker(eqx.Module):
    """
    Implements probabilistic tractography using the Global SBI Tensor model.
    """
    model: eqx.Module
    bvals: jnp.ndarray # (N,)
    bvecs: jnp.ndarray # (N, 3)
    step_size: float
    max_angle: float
    max_dirs: int # For padding input
    
    def __init__(self, model, bvals, bvecs, step_size=0.5, max_angle_deg=45.0, max_dirs=64):
        self.model = model
        self.bvals = bvals
        self.bvecs = bvecs
        self.step_size = step_size
        self.max_angle = jnp.deg2rad(max_angle_deg)
        self.max_dirs = max_dirs
        
    @eqx.filter_jit
    def track(self, seed, image_data, affine, key, max_steps=1000):
        """
        Generates a single streamline from a seed.
        
        Args:
            seed: (3,) Coordinate in VOXEL space.
            image_data: (X, Y, Z, N_meas) Signal intensity.
            affine: (4, 4) Voxel to World (placeholder).
            key: Random key.
            max_steps: Maximum length.
            
        Returns:
            streamline: (max_steps, 3) Voxel coordinates.
            valid_mask: (max_steps,) 1 if point is valid.
        """
        
        # 1. Precompute Context Features (Fixed for whole volume)
        # But we need to construct the (N_max, 6) vector per voxel.
        # N_meas <= max_dirs
        n_meas = len(self.bvals)
        pad_size = self.max_dirs - n_meas
        
        # Norm bvals
        b_norm = self.bvals / 3000.0
        
        # 2. State
        init_pos = seed
        init_dir = jnp.zeros(3)
        
        # Define Step Function
        def step_fn(carry, idx):
            pos, prev_dir, active, key = carry
            
            # --- Interpolation ---
            # Prepare indices for nearest neighbor
            xi = jnp.floor(pos).astype(jnp.int32)
            
            # Check bounds (assuming image shape known or handled by clip)
            # Safe indexing
            features_sig = image_data[
                jnp.clip(xi[0], 0, image_data.shape[0]-1),
                jnp.clip(xi[1], 0, image_data.shape[1]-1),
                jnp.clip(xi[2], 0, image_data.shape[2]-1)
            ] # (N_meas,)
            
            # Check real bounds for stopping
            in_bounds = (pos >= 0) & (pos < jnp.array(image_data.shape[:3]) - 0.5)
            is_inside = jnp.all(in_bounds)
            active = active & is_inside
            
            # --- Construct Network Input ---
            # Features: [Signal, b, vx, vy, vz, mask]
            # Construct (N_meas, 6)
            
            # Mask
            mask_meas = jnp.ones(n_meas)
            
            # Stack valid part
            # sig: (N_meas,) -> (N_meas, 1)
            feat_valid = jnp.stack([
                features_sig, 
                b_norm,
                self.bvecs[:, 0], self.bvecs[:, 1], self.bvecs[:, 2],
                mask_meas
            ], axis=-1)
            
            # Pad
            # (pad_size, 6) zeros
            feat_pad = jnp.zeros((pad_size, 6))
            
            feat_full = jnp.concatenate([feat_valid, feat_pad], axis=0) # (Max_N, 6)
            
            # Flatten
            net_input = feat_full.reshape(-1) # (Max_N * 6)
            
            # --- Query Model ---
            logits, means, sigmas = self.model(net_input)
            
            # --- Sample Tensor ---
            # Sample from Mixture
            k_cat, k_gauss, k_next = jax.random.split(key, 3)
            
            # Sample component
            # logits: (K,)
            comp_idx = jax.random.categorical(k_cat, logits)
            
            # Extract mean/sigma for that component
            # means: (K, 6)
            m = means[comp_idx]
            s = sigmas[comp_idx]
            
            # Sample value
            noise = jax.random.normal(k_gauss, (6,))
            D_sample = m + s * noise
            
            # --- Get Direction ---
            direction = get_tensor_orientation(D_sample)
            
            # Bidirectional check (align with prev)
            # If first step (idx=0), just take direction?
            # Or usually we track both ways. This is one-sided. 
            # If idx > 0, align.
            
            dot = jnp.dot(direction, prev_dir)
            align_sign = jnp.sign(dot)
            # If dot is 0 (first step), sign 0 -> direction vanishes. 
            # Fix: if idx==0, sign=1.
            factor = jnp.where(idx > 0, align_sign, 1.0)
            # If factor is 0 (ortho), just keep dir.
            factor = jnp.where(factor == 0, 1.0, factor)
            
            direction = direction * factor
            
            # Max Angle Check
            # If dot < cos(max_angle), stop?
            # cos_theta = |dot| (since we flipped)
            # Actually we already flipped, so cos_theta = dot_flipped.
            # dot_flipped is positive.
            # If cos_theta < cos(max_angle), curvature too high.
            
            # Skip check on first step
            is_curved = (jnp.abs(dot) < jnp.cos(self.max_angle)) & (idx > 0)
            active = active & (~is_curved)
            
            # Update
            new_pos = pos + direction * self.step_size * active # If inactive, stay put?
            # Actually if inactive, we just record same pos, but mask invalidates it.
            
            return (new_pos, direction, active, k_next), (pos, active) # Store CURRENT pos

        init_val = (seed, init_dir, jnp.array(True), key)
        
        final, stack = jax.lax.scan(step_fn, init_val, jnp.arange(max_steps))
        
        points, valids = stack
        
        return points, valids

