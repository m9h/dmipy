
import os
import time
import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy_jax.signal_models import gaussian_models, stick
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.optimization import CustomLMFitter
from dmipy_jax.fitting.initialization import GlobalBruteInitializer

def validate_edden():
    print("--- EDDEN Real World Validation ---")
    
    # 1. Load Data
    data_path = "data/ds004910/sub-01/ses-02/dwi"
    nii_file = os.path.join(data_path, "sub-01_ses-02_dwi.nii.gz")
    bval_file = os.path.join(data_path, "sub-01_ses-02_dwi.bval")
    bvec_file = os.path.join(data_path, "sub-01_ses-02_dwi.bvec")
    
    if not os.path.exists(nii_file):
        raise FileNotFoundError(f"Data not found at {nii_file}. Run download_data.sh first.")

    print("Loading NIfTI data...")
    img = nib.load(nii_file)
    data = img.get_fdata()
    print(f"Data shape: {data.shape}")
    
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file).T
    print(f"B-values: {bvals.shape}, B-vectors: {bvecs.shape}")

    # 2. Preprocess
    # Create mask (simple threshold)
    b0_mask = bvals < 50
    b0_vol = np.mean(data[..., b0_mask], axis=-1)
    mask = b0_vol > (0.1 * np.max(b0_vol)) # Simple brain mask
    
    # Flatten
    N_voxels = np.sum(mask)
    print(f"Brain voxels: {N_voxels}")
    
    data_flat = data[mask]
    
    # Normalize
    # We normalized by b0 per voxel usually
    b0_flat = np.mean(data_flat[:, b0_mask], axis=-1)
    data_norm = data_flat / (b0_flat[:, None] + 1e-6)
    
    # Sanitize Data
    data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)
    data_norm = np.clip(data_norm, 0, None)
    
    print(f"Data Norm Stats - Min: {np.min(data_norm)}, Max: {np.max(data_norm)}, Mean: {np.mean(data_norm)}")
    
    # 3. Setup Model
    print("Setting up model...")
    ball = gaussian_models.Ball()
    stk = stick.Stick()
    
    # Constraints
    # Ball Diff: [0, 3e-9]
    # Stick Diff: [0, 3e-9]
    # Stick params: [diff, theta, phi]
    # We need to be careful with ranges.
    ball.parameter_ranges['lambda_iso'] = (1e-10, 3e-9)
    stk.parameter_ranges['lambda_par'] = (1e-10, 3e-9)

    jax_mcm = JaxMultiCompartmentModel(models=[ball, stk])
    print("Model parameters:", jax_mcm.parameter_names)
    
    # 4. Setup Acquisition
    jax_acq = JaxAcquisition(
        bvalues=bvals,
        gradient_directions=bvecs,
        delta=None, # Usually roughly 0.02
        Delta=None  # Usually roughly 0.04
        # Note: If model doesn't use delta/Delta, it's fine. Ball/Stick generally don't for sim/fit unless specified.
    )

    # 5. Initialization
    print("Initializing...")
    t0 = time.time()
    initializer = GlobalBruteInitializer(jax_mcm)
    key = jax.random.PRNGKey(42)
    candidates = initializer.generate_random_grid(2000, key)
    
    data_jax = jnp.array(data_norm)
    
    simulator_v = jax.jit(jax.vmap(jax_mcm.model_func, in_axes=(0, None)))
    cand_preds = simulator_v(candidates, jax_acq)
    
    selector_v = jax.jit(jax.vmap(initializer.select_best_candidate, in_axes=(0, None, None)))
    
    # Chunk initialization to save memory if needed, but 120k voxels fits on GPU easily?
    # EDDEN might be larger if high res.
    # Let's chunk anyway for safety.
    BATCH_SIZE = 10000
    N_total = data_jax.shape[0]
    n_batches = int(np.ceil(N_total / BATCH_SIZE))
    
    init_params_list = []
    
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, N_total)
        batch_data = data_jax[start:end]
        batch_init = selector_v(batch_data, cand_preds, candidates)
        init_params_list.append(batch_init)
        print(f"Initialized batch {i+1}/{n_batches}")
        
    init_params = jnp.concatenate(init_params_list)
    print(f"Initialization complete in {time.time() - t0:.2f}s")
    
    # Check Init
    print(f"Init Params Stats - NaNs: {np.sum(np.isnan(init_params))}, Min: {np.min(init_params)}, Max: {np.max(init_params)}")

    # 6. Fit
    print("Fitting...")
    # Prepare Fitter
    flat_ranges = []
    scales_list = []
    for name in jax_mcm.parameter_names:
        rng = jax_mcm.parameter_ranges[name]
        card = jax_mcm.parameter_cardinality[name]
        if card == 1:
             l, h = rng
             s = h if h!=0 and not np.isinf(h) else 1.0
             scales_list.append(s)
             flat_ranges.append(rng)
        else:
             flat_ranges.extend(rng)
             for r in rng:
                  l,h=r
                  s = h if h!=0 and not np.isinf(h) else 1.0
                  scales_list.append(s)
                  
    scales = jnp.array(scales_list)
    # Increase damping for robustness on real data
    fitter = CustomLMFitter(jax_mcm.model_func, flat_ranges, scales=scales, solver_settings={'damping': 1e-2})
    fit_vmapped = jax.jit(jax.vmap(fitter.fit, in_axes=(0, None, 0)))
    
    t0 = time.time()
    
    final_params_list = []
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, N_total)
        batch_data = data_jax[start:end]
        batch_init = init_params[start:end]
        
        # Compile on first run
        res, steps = fit_vmapped(batch_data, jax_acq, batch_init)
        res.block_until_ready()
        final_params_list.append(res)
        print(f"Fitted batch {i+1}/{n_batches}")

    final_params = jnp.concatenate(final_params_list)
    fit_time = time.time() - t0
    print(f"Fit complete in {fit_time:.2f}s. Throughput: {N_total/fit_time:.2f} voxels/s")

    # 7. Save Results
    print("Saving results...")
    # Map back to volume
    # ['lambda_iso', 'mu', 'lambda_par', 'partial_volume_0', 'partial_volume_1']
    # 0: iso
    # 1,2: mu
    # 3: par
    # 4: pv0
    # 5: pv1
    
    out_dir = "examples/results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Helper to save and print stats
    def save_map(data_flat, name):
        # Stats
        d_min = np.min(data_flat)
        d_max = np.max(data_flat)
        d_mean = np.mean(data_flat)
        n_nans = np.sum(np.isnan(data_flat))
        print(f"[{name}] Min: {d_min:.2e}, Max: {d_max:.2e}, Mean: {d_mean:.2e}, NaNs: {n_nans}")
        
        vol = np.zeros(mask.shape)
        vol[mask] = data_flat
        nib.save(nib.Nifti1Image(vol, img.affine), os.path.join(out_dir, name))
        
    save_map(np.array(final_params[:, 4]), "pv_ball.nii.gz")
    save_map(np.array(final_params[:, 5]), "pv_stick.nii.gz")
    save_map(np.array(final_params[:, 0]), "diff_iso.nii.gz")
    save_map(np.array(final_params[:, 3]), "diff_par.nii.gz")
    
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    validate_edden()
