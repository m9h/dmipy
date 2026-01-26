import jax
import jax.numpy as jnp
from jax import vmap
import optax
import numpy as np
import nibabel as nib
import os
import time

from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.signal_models import g2_zeppelin
from wand_loader import WandLoader

jax.config.update("jax_enable_x64", False)

def charmed_model_fn(params, bvals, bvecs, big_delta, small_delta):
    """
    CHARMED Model: Restricted Cylinder (Intra) + Zeppelin (Extra)
    params: {theta, phi, f_intra, lambda_par_intra, lambda_par_extra, lambda_perp_extra, diameter}
    """
    theta, phi = params['theta'], params['phi']
    f_intra = params['f_intra']
    
    lambda_par_intra = params['lambda_par_intra']
    diameter = params['diameter']
    
    lambda_par_extra = params['lambda_par_extra']
    lambda_perp_extra = params['lambda_perp_extra']
    
    mu_cart = jnp.array([
        jnp.sin(theta) * jnp.cos(phi),
        jnp.sin(theta) * jnp.sin(phi),
        jnp.cos(theta)
    ])
    
    # Intra: Restricted Cylinder
    # Need to verify if RestrictedCylinder takes pre-converted mu or spherical.
    # The kernel likely takes spherical or we passed mu_cart if we call kernel directly.
    # Let's call the class __call__ which handles conversion if needed.
    # But for vmap efficiency, better to use the functional kernels if possible, 
    # OR assume the class is stateless (it is eqx.Module but depends on self params).
    # We will instantiate helper classes.
    
    # Actually, let's look at cylinder.c2_cylinder signature.
    # It takes mu_cart.
    from dmipy_jax.signal_models.cylinder_models import c2_cylinder
    signal_intra = c2_cylinder(bvals, bvecs, mu_cart, lambda_par_intra, diameter, big_delta, small_delta)
    
    # Extra: Zeppelin
    # Zeppelin kernel
    signal_extra = g2_zeppelin(bvals, bvecs, mu_cart, lambda_par_extra, lambda_perp_extra)
    
    return f_intra * signal_intra + (1 - f_intra) * signal_extra

def main():
    print("Initializing CHARMED Fit on Preprocessed WAND Data...")
    loader = WandLoader()
    data, bvals, bvecs, mask = loader.load_charmed()
    
    if data is None:
        return

    # Select Slice (Middle Z)
    nx, ny, nz, n_vol = data.shape
    z_slice = nz // 2
    
    print(f"Selecting Slice Z={z_slice}")
    slice_data = data[:, :, z_slice, :]
    slice_mask = mask[:, :, z_slice]
    
    valid_indices = np.where(slice_mask)
    voxels = slice_data[valid_indices]
    
    # Normalize by b0 (mean of b<100)
    b0_mask_indices = bvals < 100
    b0_vals = np.mean(voxels[:, b0_mask_indices], axis=1, keepdims=True)
    b0_vals[b0_vals==0] = 1.0
    voxels_norm = voxels / b0_vals
    
    # Constants
    # CHARMED protocol: Delta/delta?
    # Raw JSON said TE=59ms.
    # Typical Connectom CHARMED: Delta ~ 30-40ms, delta ~ 10-20ms.
    # We need reasonable guesses if not found.
    # Let's assume Delta=0.03, delta=0.01 for now.
    # Model sensitivity to these is moderate (affects radius estimation).
    big_delta_val = 0.035 
    small_delta_val = 0.010
    
    bvals_si = jnp.array(bvals * 1e6)
    bvecs_jax = jnp.array(bvecs)
    
    # Optimizer
    optimizer = optax.adam(learning_rate=0.005)
    
    @jax.jit
    def loss_fn(p_array, signal_target):
        # p_array: [f_intra, diameter_um, lambda_par_in, lambda_par_ex, lambda_perp_ex, theta, phi]
        
        f_intra = p_array[0]
        diameter = p_array[1] * 1e-6
        l_par_in = 2.0e-9 # Fixed intra diffusivity? Or p_array[2]*1e-9
        # Standard CHARMED often fixes intrinsic diffusivity to 1.7 or 2.0?
        # Let's fit it but constrained.
        l_par_in = p_array[2] * 1e-9
        l_par_ex = p_array[3] * 1e-9
        l_perp_ex = p_array[4] * 1e-9
        theta = p_array[5]
        phi = p_array[6]
        
        p = {
            'f_intra': f_intra,
            'diameter': diameter,
            'lambda_par_intra': l_par_in,
            'lambda_par_extra': l_par_ex,
            'lambda_perp_extra': l_perp_ex,
            'theta': theta,
            'phi': phi
        }
        
        pred = charmed_model_fn(p, bvals_si, bvecs_jax, big_delta_val, small_delta_val)
        return jnp.mean((pred - signal_target)**2)

    @jax.jit
    def train_step(opt_state, params, signal):
        grads = jax.grad(loss_fn)(params, signal)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Constraints
        # f_intra [0,1]
        params = params.at[0].set(jnp.clip(params[0], 0.01, 0.99))
        # diameter [0.1um, 20um]
        params = params.at[1].set(jnp.clip(params[1], 0.1, 20.0))
        # Diffusivities [0.1, 3.0] um^2/ms
        params = params.at[2].set(jnp.clip(params[2], 0.1, 3.0)) # intra par
        params = params.at[3].set(jnp.clip(params[3], 0.1, 3.0)) # extra par
        params = params.at[4].set(jnp.clip(params[4], 0.0, params[3])) # perp <= par
        
        return opt_state, params

    def fit_voxel(signal):
        # Init: f=0.5, d=5um, D=1.7, D_ex=1.7, D_perp=0.5, th=1.57, ph=0
        init_p = jnp.array([0.5, 5.0, 1.7, 1.7, 0.5, 1.57, 0.0])
        opt_state = optimizer.init(init_p)
        curr_p = init_p
        
        def body(i, val):
            s, p = val
            s, p = train_step(s, p, signal)
            return (s, p)
            
        final_s, final_p = jax.lax.fori_loop(0, 150, body, (opt_state, curr_p))
        return final_p

    fit_batch = jax.jit(jax.vmap(fit_voxel))
    
    # Run Batch Loop
    n_voxels = voxels_norm.shape[0]
    results = np.zeros((n_voxels, 7))
    batch_size = 500
    n_batches = int(np.ceil(n_voxels / batch_size))
    
    print(f"Fitting {n_voxels} voxels...")
    t0 = time.time()
    for i in range(n_batches):
        s = i*batch_size
        e = min((i+1)*batch_size, n_voxels)
        results[s:e] = np.array(fit_batch(jnp.array(voxels_norm[s:e])))
        
        if i % 10 == 0:
            print(f"Batch {i}/{n_batches}")
            
    print(f"Done in {time.time()-t0:.2f}s")
    
    # Save Results
    os.makedirs("results/charmed", exist_ok=True)
    headers = ['f_intra', 'diameter_um', 'lambda_par_in', 'lambda_par_ex', 'lambda_perp_ex', 'theta', 'phi']
    
    # Use affine from preprocessed nifti
    ref_img = nib.load(loader.preproc_dir + "/sub-00395_eddy_corrected_data.eddy_outlier_free_data.nii.gz")
    affine = ref_img.affine
    
    for idx, name in enumerate(headers):
        pmap = np.zeros((nx, ny))
        pmap[valid_indices] = results[:, idx]
        
        res_vol = np.zeros((nx, ny, nz))
        res_vol[:, :, z_slice] = pmap
        nib.save(nib.Nifti1Image(res_vol, affine), f"results/charmed/{name}.nii.gz")

if __name__ == "__main__":
    main()
