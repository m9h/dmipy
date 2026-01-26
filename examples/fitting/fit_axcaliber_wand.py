import jax
import jax.numpy as jnp
from jax import vmap
import optax
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import time
import os

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.cylinder import C2Cylinder, C1Stick
from dmipy_jax.distributions.distributions import DD1Gamma
from wand_loader import WandLoader

# Set JAX to use 64-bit for precision if needed, or 32 for speed
jax.config.update("jax_enable_x64", False)

def axcaliber_model_fn(params, bvals, bvecs, big_delta, small_delta, diameter_grid, diameter_pdf_norm):
    """
    params: {theta, phi, lambda_par, alpha, beta, f_intra}
    """
    # Unpack geometry
    theta, phi = params['theta'], params['phi']
    lambda_par = params['lambda_par']
    f_intra = params['f_intra']
    
    # Pre-calculated distribution passed in or params used to recalc?
    # If we fit alpha/beta, we must recalc PDF here.
    # But C2Cylinder vmap is expensive.
    # We can assume diameter grid is fixed?
    # If we fit alpha/beta, the PDF changes.
    # We can use the diameter_grid and re-evaluate Gamma PDF.
    
    alpha, beta = params['alpha'], params['beta']
    
    # Recalculate PDF
    # Gamma PDF: x^(alpha-1) * exp(-x/beta) / (beta^alpha * Gamma(alpha))
    # We use log-prob for stability? Or just basic implementation.
    # JAX gamma pdf.
    from jax.scipy.stats import gamma
    pdf = gamma.pdf(diameter_grid, a=alpha, scale=beta)
    # Normalize
    pdf_norm = pdf / (jnp.sum(pdf) + 1e-12)
    
    # Intra-axonal Signal Integration
    mu_cart = jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])
    
    # We pre-calculate or vmap the Cylinder response for each diameter in the GRID.
    # Since big_delta varies per shell, we can't pre-calculate easily unless we group by shell.
    # But JAX vmap can handle it.
    
    # Define single cylinder call
    def signal_for_d(d):
        return C2Cylinder()(bvals, bvecs, 
                          mu=mu_cart, 
                          lambda_par=lambda_par,
                          diameter=d,
                          big_delta=big_delta,
                          small_delta=small_delta)
    
    # vmap over diameters
    # diameter_grid is (N_d,)
    signals_intra_d = vmap(signal_for_d)(diameter_grid) # (N_d, N_acq)
    
    # Integrate: sum(signal(d) * pdf(d))
    signal_intra = jnp.dot(pdf_norm, signals_intra_d) # (N_acq,)
    
    # Extra-axonal (Stick)
    signal_extra = C1Stick()(bvals, bvecs, mu=mu_cart, lambda_par=lambda_par)
    
    # Combine
    return f_intra * signal_intra + (1 - f_intra) * signal_extra

def main():
    print("Initializing AxCaliber Fit on WAND Data...")
    loader = WandLoader()
    data, bvals, bvecs, big_deltas, small_deltas = loader.load_axcaliber()
    
    # Select a slice (Middle axial slice)
    # data shape: (X, Y, Z, N)
    nx, ny, nz, n_vol = data.shape
    z_slice = nz // 2
    
    print(f"Selecting Slice Z={z_slice} (Shape: {nx}x{ny})")
    slice_data = data[:, :, z_slice, :]
    
    # Mask: Simple threshold
    b0_mask = bvals < 100
    b0_img = np.mean(slice_data[..., b0_mask], axis=-1)
    mask = b0_img > (0.1 * np.max(b0_img)) # 10% threshold
    
    print(f"Mask has {np.sum(mask)} voxels.")
    
    # Normalize signals
    data_norm = np.zeros_like(slice_data)
    # Need to normalize by b0, but AxCaliber has different TEs? 
    # No, we verified TE=80ms constant. So simple b0 norm is fine.
    b0_val = b0_img[..., None]
    b0_val[b0_val==0] = 1.0
    data_norm = slice_data / b0_val
    
    # Convert to JAX arrays (flatten masked voxels for batch processing)
    valid_indices = np.where(mask)
    voxels = data_norm[valid_indices] # (N_vox, N_acq)
    
    # Batch Processing
    batch_size = 500 # Adjust for GPU memory
    n_voxels = voxels.shape[0]
    n_batches = int(np.ceil(n_voxels / batch_size))
    
    print(f"Processing {n_voxels} voxels in {n_batches} batches...")
    
    # Global constants
    # Convert bvals to s/m^2 (from s/mm^2 usually)
    # Checked bval: ~6000? No, AxCaliber max was like 4400.
    # WAND bvals are in s/mm^2.
    bvals_si = jnp.array(bvals * 1e6)
    bvecs_jax = jnp.array(bvecs)
    big_deltas_jax = jnp.array(big_deltas)
    small_deltas_jax = jnp.array(small_deltas)
    
    # Diameter Grid for Integration (0.1um to 20um)
    diameter_grid = jnp.linspace(0.1e-6, 10e-6, 30) # 30 steps
    
    # Loss Function
    @jax.jit
    def loss_fn(params_array, signal_target):
        # params_array: [alpha, beta_um, f_intra, theta, phi]
        # lambda_par fixed 1.7e-9? Or fit? Let's fit it. [..., lambda_um]
        
        alpha = params_array[0]
        beta = params_array[1] * 1e-6
        f_intra = params_array[2]
        theta = params_array[3]
        phi = params_array[4]
        lambda_par = 1.7e-9 # Fixed for stability or params_array[5] * 1e-9?
        
        p = {
            'alpha': alpha, 'beta': beta, 'f_intra': f_intra,
            'theta': theta, 'phi': phi, 'lambda_par': lambda_par
        }
        
        pred = axcaliber_model_fn(p, bvals_si, bvecs_jax, big_deltas_jax, small_deltas_jax, diameter_grid, None)
        return jnp.mean((pred - signal_target)**2)
    
    # Batch Fit Function
    # We use vmap over voxels? No, too big. Loop batches, vmap inside.
    
    optimizer = optax.adam(learning_rate=0.01)
    
    @jax.jit
    def train_step(opt_state, params, signal):
        grads = jax.grad(loss_fn)(params, signal)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Constraints
        # alpha > 0.5 (Gamma instability near 0)
        params = params.at[0].set(jnp.maximum(params[0], 0.5))
        # beta_um > 0.1 (100 nm min)
        params = params.at[1].set(jnp.maximum(params[1], 0.1))
        # f_intra [0.1, 0.99]
        params = params.at[2].set(jnp.clip(params[2], 0.1, 0.99))
        
        return opt_state, params
    
    # Vmapped training for a batch?
    # Actually, standard approach: vmap the train_step over a batch of voxels?
    # But train_step updates params.
    # We want to fit each voxel INDEPENDENTLY.
    # So we vmap the ENTIRE optimization loop? 
    # Or simplified: use vmap(value_and_grad) but we need separate params for each voxel.
    
    # Defined fitted function for ONE voxel
    def fit_voxel(signal):
        # Initial Guess: alpha=4, beta=0.5, f=0.5, theta=1.57, phi=0
        # Determine rough direction from DTI? Too slow. Use fixed or heuristic.
        # Just start at x-axis?
        init_params = jnp.array([4.0, 0.5, 0.5, 1.57, 0.0])
        opt_state = optimizer.init(init_params)
        
        curr_params = init_params
        
        # Fixed number of iterations
        def body(i, val):
            state, p = val
            state, p = train_step(state, p, signal)
            return (state, p)
            
        final_state, final_params = jax.lax.fori_loop(0, 100, body, (opt_state, curr_params))
        return jnp.where(jnp.isnan(final_params).any(), init_params, final_params)

    fit_batch = jax.jit(jax.vmap(fit_voxel))
    
    # Results containers
    # [alpha, beta_um, f_intra, theta, phi]
    out_headers = ['alpha', 'beta_um', 'f_intra', 'theta', 'phi']
    results = np.zeros((n_voxels, 5))
    
    t0 = time.time()
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_voxels)
        batch_signals = jnp.array(voxels[start:end])
        
        batch_res = fit_batch(batch_signals)
        results[start:end] = np.array(batch_res)
        
        if i % 10 == 0:
            print(f"Batch {i}/{n_batches} done.")
            
    print(f"Fitting done in {time.time()-t0:.2f}s")
    
    # Debug: Check results
    print(f"Results Mean: {np.mean(results, axis=0)}")
    print(f"Results Max: {np.max(results, axis=0)}")
    
    # Reconstruct Images
    out_maps = {}
    for idx, name in enumerate(out_headers):
        pmap = np.zeros((nx, ny))
        pmap[valid_indices] = results[:, idx]
        out_maps[name] = pmap
        
    # Validating derived param: Mean Diameter
    # mean = 2 * alpha * beta
    mean_d = 2 * out_maps['alpha'] * out_maps['beta_um']
    out_maps['mean_diameter_um'] = mean_d
    
    # Save NIfTIs
    print("Saving maps...")
    os.makedirs("results/axcaliber", exist_ok=True)
    affine = nib.load(loader.dwi_dir + "/sub-00395_ses-02_acq-AxCaliber1_dir-AP_part-mag_dwi.nii.gz").affine
    
    for name, data_map in out_maps.items():
        # Save as 3D (one slice populated)
        res_vol = np.zeros((nx, ny, nz))
        res_vol[:, :, z_slice] = data_map
        nib.save(nib.Nifti1Image(res_vol, affine), f"results/axcaliber/{name}.nii.gz")
        
    print("Done.")

if __name__ == "__main__":
    main()
