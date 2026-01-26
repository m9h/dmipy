
import os
import jax
import jax.numpy as jnp
# Monkeypatch for JAX 0.6.0 compatibility with older jaxopt
jax.tree_map = jax.tree_util.tree_map
import numpy as np
import nibabel as nib
import jaxopt
from dmipy_jax.models.qmt import QMTSPGR, QMTParameters
from wand_loader import WandLoader

def run_fitting():
    print("Initializing qMT Fitting Pipeline...")
    
    # 1. Load Data
    # ---------------------------------------------------------
    # Using Sub-00395 from WAND
    loader = WandLoader(base_dir="data/wand/sub-00395/ses-02")
    data, offsets, mt_powers, ex_flips_meas, TRs = loader.load_qmt()
    
    if data is None:
        print("Data load failed.")
        return

    # Check Dimensions
    print(f"Data Shape: {data.shape}")
    X, Y, Z, N_meas = data.shape
    
    # Masking ( Simple threshold on mean signal )
    mean_sig = np.mean(data, axis=-1)
    mask = mean_sig > (0.1 * np.max(mean_sig)) # generous mask
    print(f"Mask counts: {np.sum(mask)} out of {mask.size} voxels.")

    # 2. Define Model & Loss
    # ---------------------------------------------------------
    model = QMTSPGR()
    
    # Fixed Parameters / Assumptions
    # T1_b is often fixed to 1s
    # T2_b determines lineshape width, often fixed ~10us
    FIXED_T1_B = 1.0
    FIXED_T2_B = 0.000012 # 12us
    
    # Protocol Arrays (Device Arrays)
    p_offsets = jnp.array(offsets)
    p_mt_powers = jnp.array(mt_powers)
    p_tr = float(TRs[0]) # Assume constant TR for now?
    p_ex_flip = float(ex_flips_meas[0]) # Assume constant Excitation?
    # Note: If TR or ExFlip vary per measurement, we need to pass arrays to model wrapper.
    
    # QMT Model wrapper for optimization
    # Params vector theta = [f_bound, k_fb, T1_f, T2_f]
    # We fit log parameters to enforce positivity? Or use Bounded optimization.
    
    def forward_model(theta):
        f, k, T1f, T2f = theta
        
        # Constraints:
        # f in [0, 1] -> Sigmoid?
        # k > 0 -> Softplus?
        
        params = QMTParameters(
            f_bound=f,
            k_fb=k,
            T1_f=T1f,
            T2_f=T2f,
            T1_b=FIXED_T1_B,
            T2_b=FIXED_T2_B
        )
        
        # Scan over measurements
        # QMTSPGR doesn't vectorize over protocol automatically yet.
        # We define a mapped function.
        
        def single_meas(offset, mt_pow):
            return model(params, p_tr, p_ex_flip, mt_pow, offset, mt_pulse_duration=0.010)
            
        signals = jax.vmap(single_meas)(p_offsets, p_mt_powers)
        return signals

    def loss_fn(theta, y_true, **kwargs):
        y_pred = forward_model(theta)
        # Residuals for LM
        return y_pred - y_true 

    # 3. Optimization Strategy (Levenberg-Marquardt)
    # ---------------------------------------------------------
    # Solver: jaxopt.LevenbergMarquardt
    # Requires function f(x) -> residuals
    # Note: damping can be passed to run() or implicit?
    # Actually jaxopt.LevenbergMarquardt takes 'damping_parameter' in run(), or simply solves strict LM.
    # We will use default initialization.
    
    solver = jaxopt.LevenbergMarquardt(residual_fun=loss_fn, maxiter=20)
    
    # Initialization
    # f=0.1, k=2.0, T1f=1.0, T2f=0.05
    init_theta = jnp.array([0.1, 2.0, 1.0, 0.05])
    
    # Voxelwise Wrapper
    def fit_voxel(signal):
        # Scale signal?
        # Model returns magnetization (M0=1).
        # We need to fit a global scale M0?
        # y_true = M0 * y_pred
        # Or normalize signal by MT-off?
        
        # Normalize by max signal approx M0?
        s_max = jnp.max(signal)
        y_norm = signal / (s_max + 1e-9)
        
        # Run Solver
        # We fit [f, k, T1f, T2f]. 
        # damping_parameter=1e-3
        
        sol = solver.run(init_theta, y_true=y_norm, damping_parameter=1e-3)
        return sol.params, sol.state.iter_num
        
    print("JIT Compiling Solver...")
    fit_voxel_jit = jax.jit(fit_voxel)
    
    # Test on center voxel
    center = (X//2, Y//2, Z//2)
    sig_c = jnp.array(data[center])
    print(f"Running Test Voxel {center}...")
    p_est, nit = fit_voxel_jit(sig_c)
    print(f"Est: f={p_est[0]:.3f}, k={p_est[1]:.3f}, T1f={p_est[2]:.3f}, T2f={p_est[3]:.3f} (iters={nit})")

    # 4. Batch Processing
    # ---------------------------------------------------------
    print("Running Whole Volume Fit...")
    # Flat mask
    mask_flat = mask.reshape(-1)
    data_flat = data.reshape(-1, N_meas)
    
    # Run only on mask
    data_masked = jnp.array(data_flat[mask_flat])
    
    # Batch using vmap (chunked if needed)
    # Depending on GPU memory. 
    # Batch size 10000?
    
    batch_size = 5000
    n_vox = data_masked.shape[0]
    n_batches = int(np.ceil(n_vox / batch_size))
    
    results = []
    
    fit_batch = jax.vmap(fit_voxel)
    
    for i in range(n_batches):
        if i % 10 == 0:
            print(f"Batch {i}/{n_batches}")
        idx0 = i * batch_size
        idx1 = min((i+1)*batch_size, n_vox)
        batch = data_masked[idx0:idx1]
        
        # Convert to JAX
        batch_j = jnp.array(batch)
        p_out, _ = fit_batch(batch_j)
        # Force wait
        p_out.block_until_ready()
        results.append(np.array(p_out))
        
    results_flat = np.concatenate(results, axis=0) # (N_masked, 4)
    
    # 5. Reconstruct Volume
    # ---------------------------------------------------------
    # Output maps
    out_vol = np.zeros((X*Y*Z, 4), dtype=np.float32)
    out_vol[mask_flat] = results_flat
    out_vol = out_vol.reshape(X, Y, Z, 4)
    
    # Save
    out_dir = "results/fit_qmt"
    os.makedirs(out_dir, exist_ok=True)
    
    param_names = ["f_bound", "k_fb", "T1_f", "T2_f"]
    affine = np.eye(4) # TODO: Use real affine from WandLoader
    
    for i, name in enumerate(param_names):
        print(f"Saving {name}...")
        img = nib.Nifti1Image(out_vol[..., i], affine)
        nib.save(img, os.path.join(out_dir, f"qmt_{name}.nii.gz"))
        
    print("Done.")

if __name__ == "__main__":
    run_fitting()
