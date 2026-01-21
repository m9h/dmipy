import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.reconstruction.response import ResponseEstimator
from dmipy_jax.utils.spherical_harmonics import sh_basis_real, cart2sphere

def generate_single_fiber_field(bvecs, bval, response_eigenvalues):
    # Generates a field of single fiber voxels with random orientations.
    # N_voxels = 500
    N_vox = 500
    
    key = jax.random.PRNGKey(42)
    # Random orientations
    dirs = jax.random.normal(key, (N_vox, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Rician noise sigma
    sigma = 0.02
    
    # Generate signal S = exp(-b * gT D g)
    # Tensor D = R Lambda R^T
    # Lambda = diag(e1, e2, e2)
    e1, e2, e3 = response_eigenvalues
    Lambda = jnp.diag(jnp.array([e1, e2, e3]))
    
    # Helper to generate D for a direction v
    def get_D(v):
        # Rotate Z to v
        # Simple construction:
        z = jnp.array([0., 0., 1.])
        # If v ~ z, R=I.
        # ... Reuse rotation logic or just synthetic formula:
        # D = e2 I + (e1-e2) v v^T
        return e2 * jnp.eye(3) + (e1 - e2) * jnp.outer(v, v)
        
    Ds = jax.vmap(get_D)(dirs) # (500, 3, 3)
    
    # Signal for each voxel
    # S_i = exp( -b * trace(D_i G_i) ) ? No.
    # S_i(g) = exp( -b * g^T D_i g )
    
    # bvecs (N_meas, 3)
    # S (500, N_meas)
    
    def predict_voxel(D):
         # g^T D g
         # (N_meas, 3) @ (3, 3) @ (3, N_meas) -> diagonals
         # faster: sum( (g @ D) * g, axis=1)
         
         gD = jnp.dot(bvecs, D) # (N_meas, 3)
         quad = jnp.sum(gD * bvecs, axis=1)
         return jnp.exp(-bval * quad)
         
    signals = jax.vmap(predict_voxel)(Ds)
    
    # Add noise
    noise1 = jax.random.normal(key, signals.shape) * sigma
    noise2 = jax.random.normal(jax.random.PRNGKey(99), signals.shape) * sigma
    noisy_signals = jnp.sqrt((signals + noise1)**2 + noise2**2)
    
    return noisy_signals

def main():
    print("Verifying Response Estimation...")
    
    # Setup
    bval = 3000.0 # High b-value for sharp response
    evals_wm = (1.7e-3, 0.2e-3, 0.2e-3)
    
    # 64 directions
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (64, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = jnp.full(64, bval)
    
    # Generate Data
    print("Generating Synthetic Volume (500 single-fiber voxels)...")
    data = generate_single_fiber_field(bvecs, bval, evals_wm)
    
    # Estimate
    print("Running ResponseEstimator (Tournier)...")
    estimator = ResponseEstimator(sh_order=8)
    
    # Create dummy mask (all valid)
    mask = jnp.ones(500)
    
    # Estimate
    wm_coeffs, gm, csf = estimator.estimate(data, bvals, bvecs, mask)
    
    print(f"Estimated WM SH Coefficients (first 5 zonal): {wm_coeffs[:5]}")
    
    # Ground Truth Zonal SH
    # Calculate response for D aligned with Z
    # D_gt = diag(1.7, 0.2, 0.2)
    # Simulate aligned signal and fit
    theta = jnp.linspace(0, jnp.pi, 100)
    phi = jnp.zeros_like(theta)
    s_aligned = jnp.exp(-bval * (evals_wm[0]*jnp.cos(theta)**2 + evals_wm[1]*jnp.sin(theta)**2))
    
    # Fit SH
    Y = sh_basis_real(theta, phi, 8)
    # Zonal only
    m0_indices = []
    curr = 0
    for l in range(0, 9, 2):
        m0_indices.append(curr + l)
        curr += (2 * l + 1)
    Y_zonal = Y[:, jnp.array(m0_indices)]
    
    c_gt = jnp.linalg.lstsq(Y_zonal, s_aligned, rcond=None)[0]
    
    print(f"Ground Truth coefficients: {c_gt}")
    
    # Error
    err = jnp.mean(jnp.abs(wm_coeffs - c_gt))
    print(f"Mean Absolute Error: {err:.5f}")
    
    if err < 0.05:
         print("SUCCESS: Estimation matches ground truth.")
    else:
         print("FAILURE: High estimation error.")

if __name__ == "__main__":
    main()
