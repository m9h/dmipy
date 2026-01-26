
import jax
import jax.numpy as jnp
import numpy as np
import time

from dmipy_jax.models.mcdespot import McDESPOT, McDESPOTParameters

# --- Native JAX Solver (Replacing broken SCICO dependency) ---
# The installed scico version clashes with JAX 0.4+ (DeviceArray removal).
# We implement Projected Gradient Method (PGM) manually for TV Regularization.

def create_phantom(shape=(64, 64)):
    """
    Create a 2D Phantom with structure for MWF.
    """
    # Background
    mwf = jnp.zeros(shape)
    T1_m = jnp.zeros(shape) + 500.0
    T2_m = jnp.zeros(shape) + 20.0
    T1_ie = jnp.zeros(shape) + 1000.0
    T2_ie = jnp.zeros(shape) + 80.0
    
    # Circle (Simulation of Brain/ROI)
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, shape[1]), jnp.linspace(-1, 1, shape[0]))
    mask = (x**2 + y**2) < 0.6
    
    # Graded MWF inside mask (Smooth variation)
    mwf = jnp.where(mask, 0.15 + 0.05 * jnp.sin(3*x), 0.0)
    
    # Sharp Block (Lesion or Structure)
    block_mask = (jnp.abs(x) < 0.2) & (jnp.abs(y) < 0.2)
    mwf = jnp.where(block_mask, 0.05, mwf)
    
    return mwf, mask, T1_m, T2_m, T1_ie, T2_ie

def run_tv_recon_demo():
    print(f"--> Starting McDESPOT TV Reconstruction (Native JAX PGM)...")
    
    # 1. Setup Data
    N = 64
    mwf_true, mask, T1_m, T2_m, T1_ie, T2_ie = create_phantom((N, N))
    
    # McDESPOT Model
    model = McDESPOT()
    TR = 5.0
    
    # Protocol
    spgr_alphas = jnp.deg2rad(jnp.array([5.0, 15.0])) # Reduced set for speed
    ssfp_alphas = jnp.deg2rad(jnp.array([10.0, 40.0]))
    
    # Helper to run forward model on maps
    @jax.jit
    def forward_map(mwf_map):
        flat_mwf = mwf_map.ravel()
        # Flatten fixed maps
        flat_T1m = T1_m.ravel()
        flat_T2m = T2_m.ravel()
        flat_T1ie = T1_ie.ravel()
        flat_T2ie = T2_ie.ravel()
        
        def single_voxel(f, t1m, t2m, t1ie, t2ie):
            p = McDESPOTParameters(f, t1m, t2m, t1ie, t2ie, 0.0)
            
            s_spgr = jax.vmap(lambda a: model(p, 'SPGR', TR, a))(spgr_alphas)
            s_ssfp0 = jax.vmap(lambda a: model(p, 'SSFP', TR, a, 0.0))(ssfp_alphas)
            s_ssfp180 = jax.vmap(lambda a: model(p, 'SSFP', TR, a, jnp.pi))(ssfp_alphas)
            
            return jnp.concatenate([s_spgr, s_ssfp0, s_ssfp180])

        Y = jax.vmap(single_voxel)(flat_mwf, flat_T1m, flat_T2m, flat_T1ie, flat_T2ie)
        return Y # Shape (N*N, N_measurements)

    print("--> Generatng Synthetic Data...")
    y_true = forward_map(mwf_true)
    
    # Add Noise
    noise_sigma = 0.01
    key = jax.random.PRNGKey(0)
    y_noisy = y_true + noise_sigma * jax.random.normal(key, y_true.shape)
    
    # 2. Define Loss with Huber TV
    lambda_tv = 2.0e-2 # Tune manually
    huber_delta = 0.01
    
    @jax.jit
    def huber_tv(x):
        # Finite Differences
        diff_x = x[1:, :] - x[:-1, :]
        diff_y = x[:, 1:] - x[:, :-1]
        
        # Pad to keep shape? Or just sum valid
        # Huber: sqrt(d^2 + delta^2) - delta
        tv_x = jnp.sum(jnp.sqrt(diff_x**2 + huber_delta**2) - huber_delta)
        tv_y = jnp.sum(jnp.sqrt(diff_y**2 + huber_delta**2) - huber_delta)
        return tv_x + tv_y

    @jax.jit
    def total_loss(x):
        # MSE
        diff = forward_map(x) - y_noisy
        mse = 0.5 * jnp.sum(diff**2)
        # TV
        tv = lambda_tv * huber_tv(x)
        return mse + tv

    # 3. Solver: Projected Gradient Descent (FISTA/Nesterov possible, simple GD for demo)
    # Projection: mwf in [0, 1]
    
    @jax.jit
    def update_step(x, lr):
        grads = jax.grad(total_loss)(x)
        x_new = x - lr * grads
        # Projection
        x_new = jnp.clip(x_new, 0.0, 1.0)
        return x_new

    # Loop
    x_est = jnp.zeros((N, N)) + 0.1 # Initial Guess
    lr = 0.01
    
    print(f"--> Running PGM Solver (Lambda TV={lambda_tv})...")
    start_t = time.time()
    for i in range(200):
        # Basic line search or decay? Fixed LR for demo
        x_est = update_step(x_est, lr)
        
        if i % 20 == 0:
            l = total_loss(x_est)
            print(f"Iter {i}: Loss {l:.5f}")
            
    print(f"--> Solver Finished in {time.time() - start_t:.2f}s")
    
    # 4. Validation
    mse_recon = jnp.mean((x_est - mwf_true)**2)
    print(f"\nReconstruction MSE: {mse_recon:.6f}")
    
    # Check "Lesion" contrast
    center_val = x_est[32, 32]
    bg_val = x_est[10, 10]
    
    print(f"Lesion Value: {center_val:.3f} (True: 0.05)")
    print(f"Background Val: {bg_val:.3f} (True: ~0.0)")
    
    if jnp.abs(center_val - 0.05) < 0.05:
         print("SUCCESS: Lesion structure preserved.")
    else:
         print("WARNING: Contrast may be washed out.")

if __name__ == "__main__":
    run_tv_recon_demo()

