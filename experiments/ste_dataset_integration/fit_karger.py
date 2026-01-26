import jax
import jax.numpy as jnp
import optax
import numpy as np
import nibabel as nib
import json
import os
import time
from dmipy_jax.models.ste_karger import KargerExchangeModel
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

# Enable float64 for matrix exp stability
jax.config.update("jax_enable_x64", True)

def load_data(manifest_path, key):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    path = manifest.get(key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Data not found for key: {key}")
        
    print(f"Loading {key} from {path}")
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    
    # Load bvals/bvecs (Assume standard location relative to image)
    # Usually in same dir
    base_dir = os.path.dirname(path)
    bvals_path = os.path.join(base_dir, "bvals.txt")
    bvecs_path = os.path.join(base_dir, "bvecs.txt")
    
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path).T
    
    # Normalize bvecs
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    
    return jnp.array(data), jnp.array(bvals), jnp.array(bvecs), affine

def fit_volume(data, bvals, bvecs, affine, output_prefix):
    print(f"Fitting volume shape: {data.shape}")
    
    # Setup Model
    # Important: NITRC STE data usually has fixed TM?
    # We need to look up TM from protocol or assume.
    # Looking at directory: STE00_ExVivo/STE
    # Assuming TM is constant for the STE block.
    # Implementation Note: We'll assume TM=0.2s for now as placeholder or 
    # infer if multiple shells have diff TMs. 
    # Usually bvals.txt doesn't have TM.
    # For this exercise we use a single scalar TM broadcasted.
    TM_VAL = 0.1 # Placeholder: 100ms
    
    
    # Need to construct acquisition scheme object (simple wrapper)
    class Scheme:
        bvalues = bvals
        mixing_time = jnp.full(bvals.shape, TM_VAL)
        
    scheme = Scheme()
    model = KargerExchangeModel()
    
    # Loss Function
    def loss_fn(params, signal_obs):
        # params is array [D_intra, D_extra, f_intra, exchange_time]
        p_dict = {
            'D_intra': params[0],
            'D_extra': params[1],
            'f_intra': params[2],
            'exchange_time': params[3]
        }
        signal_est = model(scheme, p_dict)
        return jnp.mean((signal_est - signal_obs)**2)

    # Optimizer (Adam for simplicity in JAX, or Lwebfgs if available, 
    # but fit_karger usually implies voxelwise optimization)
    # We will use jax.scipy.optimize.minimize (BFGS) if possible, 
    # but it maps per voxel.
    
    import jax.scipy.optimize
    
    @jax.jit
    def fit_voxel(signal_obs):
        # Initial guess
        # Di=2e-9, De=1e-9, f=0.5, tau=0.1
        x0 = jnp.array([2.0e-3, 1.0e-3, 0.5, 0.1]) # Using um^2/ms units? NO, we used SI in model (1e-9)
        # SI Units:
        x0 = jnp.array([2.0e-9, 1.0e-9, 0.5, 0.1])
        
        # Bounds? jax.scipy.optimize.minimize doesn't maintain bounds easily in BFGS.
        # Use simple projected gradient or parameter transform.
        # Transform: D = exp(p), f = sigmoid(p)
        
        def loss_transformed(p_trans, s_obs):
            p_orig = jnp.array([
                jnp.exp(p_trans[0]), # Di > 0
                jnp.exp(p_trans[1]), # De > 0
                jax.nn.sigmoid(p_trans[2]), # 0 < f < 1
                jnp.exp(p_trans[3]) # tau > 0
            ])
            return loss_fn(p_orig, s_obs)
            
        x0_trans = jnp.array([
            jnp.log(2.0e-9),
            jnp.log(1.0e-9),
            0.0, # sigmoid(0)=0.5
            jnp.log(0.1)
        ])
        
        res = jax.scipy.optimize.minimize(loss_transformed, x0_trans, args=(signal_obs,), method='BFGS')
        
        p_final_trans = res.x
        p_final = jnp.array([
            jnp.exp(p_final_trans[0]),
            jnp.exp(p_final_trans[1]),
            jax.nn.sigmoid(p_final_trans[2]),
            jnp.exp(p_final_trans[3])
        ])
        
        return p_final, res.fun

    print("Compiling fit...")
    # Select small ROI to be fast?
    # Or just mask.
    mask = data[..., 0] > 1e-3 # Simple background masking
    
    # Flatten spatial dims
    X, Y, Z, N = data.shape
    data_flat = data.reshape(-1, N)
    mask_flat = mask.reshape(-1)
    
    # We'll just fit a subset (center slice) for demo speed
    center_z = Z // 2
    print(f"Flattening and selecting Slice {center_z}...")
    
    # Actually, SiliconWeaver wants speed. Let's vmap the whole slice.
    slice_data = data[:, :, center_z, :].reshape(-1, N)
    
    print("Fitting Center Slice...")
    t0 = time.time()
    params_slice, losses_slice = jax.vmap(fit_voxel)(slice_data)
    # params_slice: (Voxels, 4)
    jax.block_until_ready(params_slice)
    print(f"Fit complete in {time.time() - t0:.2f}s")
    
    # Reshape
    # params: (X*Y, 4) -> (X, Y, 4)
    # Full volume placeholder
    out_vol = np.zeros((X, Y, Z, 4))
    out_vol[:, :, center_z, :] = np.array(params_slice).reshape(X, Y, 4)
    
    # Save maps
    names = ['D_intra', 'D_extra', 'f_intra', 'exchange_time']
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    for i, name in enumerate(names):
        vol = out_vol[..., i]
        path = f"{output_prefix}_{name}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine), path)
        print(f"Saved {path}")

def main():
    manifest_path = "experiments/ste_dataset_integration/data_manifest.json"
    
    # Process Ex-Vivo
    print("--- Processing Ex-Vivo ---")
    data, bvals, bvecs, affine = load_data(manifest_path, 'ex_vivo')
    fit_volume(data, bvals, bvecs, affine, "experiments/ste_dataset_integration/results/ex_vivo")
    
    # Process In-Vivo
    print("--- Processing In-Vivo ---")
    data, bvals, bvecs, affine = load_data(manifest_path, 'in_vivo')
    fit_volume(data, bvals, bvecs, affine, "experiments/ste_dataset_integration/results/in_vivo")

if __name__ == "__main__":
    main()
