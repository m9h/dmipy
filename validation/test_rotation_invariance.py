
import jax
import jax.numpy as jnp
import numpy as np
import os
from dmipy_jax.core.invariants import compute_invariants
from dmipy_jax.cylinder import C2Cylinder
from scipy.spatial.transform import Rotation as R

def normalize_bvecs(bvecs):
    norms = jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    return jnp.where(norms == 0, 0, bvecs / norms)

def rotate_bvecs(bvecs, angles):
    """
    Rotates b-vectors using Euler angles (degrees).
    angles: (alpha, beta, gamma)
    """
    rot = R.from_euler('zyx', angles, degrees=True)
    # Apply rotation to bvecs
    # bvecs shape (N, 3)
    # v_new = R * v_old
    bvecs_rot = rot.apply(bvecs)
    return jnp.array(bvecs_rot)

def test_rotation_invariance():
    print("Testing Rotation Invariance...")
    
    # 1. Setup Single Voxel Model
    # Region A properties
    lambda_par = 1.7e-9 
    mu_0 = jnp.array([0., 0.]) # Z-axis
    diam = 4.0e-6 
    big_delta = 30e-3
    small_delta = 10e-3
    
    cyl = C2Cylinder()
    
    # 2. Gradient Table
    # Simple shell
    bval = 3000
    n_dirs = 64
    # quick sphere code from phantom script or just random
    rng = np.random.default_rng(42)
    dirs_random = rng.normal(size=(n_dirs, 3))
    dirs_random /= np.linalg.norm(dirs_random, axis=1)[:, None]
    
    bvecs_0 = jnp.array(dirs_random)
    bvals_0 = jnp.array([bval] * n_dirs)
    
    # 3. Simulate Signal 0 (Reference)
    params = {
        'lambda_par': lambda_par,
        'diameter': diam,
        'mu': mu_0,
        'big_delta': big_delta,
        'small_delta': small_delta
    }
    
    # Helper to call model
    # C2Cylinder takes cartesian mu internally if we passed direct cartesian in previous attempts, 
    # but here let's stick to the interface. 
    # Wait, my phantom script passed mu=(theta, phi). C2Cylinder converted it.
    
    # Signal 0
    # Note: Model simulates basic physics. 
    # If we rotate the gradients, we simulate what the scanner sees if the gradients were applied in new directions.
    # The Fiber is FIXED at mu_0.
    
    sig_0 = cyl(bvals_0, bvecs_0, **params)
    
    # 4. Compute Invariants 0
    # Reshape signal to (1, N_dirs) for function
    sig_0_reshaped = sig_0[None, :]
    inv_0 = compute_invariants(sig_0_reshaped, bvecs_0, max_order=6)
    
    print(f"Invariants 0 (first 5): {inv_0[0, :5]}")
    
    # 5. Rotate Gradients
    alpha, beta, gamma = 30, 45, 60
    bvecs_rot = rotate_bvecs(bvecs_0, [alpha, beta, gamma])
    
    # 6. Simulate Signal Rotated
    # Same Fiber, Rotated Gradients
    sig_rot = cyl(bvals_0, bvecs_rot, **params)
    
    # 7. Compute Invariants Rotated
    sig_rot_reshaped = sig_rot[None, :]
    inv_rot = compute_invariants(sig_rot_reshaped, bvecs_rot, max_order=6)
    
    print(f"Invariants Rot (first 5): {inv_rot[0, :5]}")
    
    # 8. Comparison
    mse = jnp.mean((inv_0 - inv_rot)**2)
    max_diff = jnp.max(jnp.abs(inv_0 - inv_rot))
    
    print(f"MSE: {mse:.2e}")
    print(f"Max Diff: {max_diff:.2e}")
    
    assert max_diff < 1e-6, f"Rotation Test Failed! Max Diff {max_diff} > 1e-6"
    print("SUCCESS: Invariants match within machine precision.")

if __name__ == "__main__":
    test_rotation_invariance()
