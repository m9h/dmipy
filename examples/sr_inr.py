import os
import jax
import jax.numpy as jnp
import equinox as eqx
import nibabel as nib
import numpy as np
import optax
from scipy.ndimage import zoom

class SIRENLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    omega: float
    
    def __init__(self, in_features, out_features, key, omega=30.0, is_first=False):
        w_init = jax.nn.initializers.uniform(
            scale=1 / in_features if is_first else np.sqrt(6 / in_features) / omega
        )
        self.weight = w_init(key, (out_features, in_features))
        self.bias = jax.numpy.zeros(out_features)
        self.omega = omega
        
    def __call__(self, x):
        return jnp.sin(self.omega * (self.weight @ x + self.bias))

class CoordinateMLP(eqx.Module):
    layers: list
    final_layer: eqx.nn.Linear
    
    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            SIRENLayer(3, 64, keys[0], is_first=True),
            SIRENLayer(64, 64, keys[1]),
            SIRENLayer(64, 64, keys[2])
        ]
        self.final_layer = eqx.nn.Linear(64, 1, key=keys[3])
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

def load_data(sub_id="sub-01"):
    base_dir = f"/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/{sub_id}"
    dwi_path = os.path.join(base_dir, "dwi", "dwi.nii.gz")
    t1_path = os.path.join(base_dir, "anat", "t1.nii.gz")
    return nib.load(dwi_path), nib.load(t1_path)

def run_inr():
    dwi_img, t1_img = load_data()
    dwi_data = dwi_img.get_fdata()[..., 0] # b0 only for demo
    
    # Create coordinate grids
    # Low res grid (Training)
    shape_lr = dwi_data.shape
    x_lr = np.linspace(-1, 1, shape_lr[0])
    y_lr = np.linspace(-1, 1, shape_lr[1])
    z_lr = np.linspace(-1, 1, shape_lr[2])
    coords_lr = np.stack(np.meshgrid(x_lr, y_lr, z_lr, indexing='ij'), axis=-1).reshape(-1, 3)
    values_lr = dwi_data.reshape(-1, 1)
    
    # High res grid (Inference)
    t1_shape = t1_img.shape
    x_hr = np.linspace(-1, 1, t1_shape[0])
    y_hr = np.linspace(-1, 1, t1_shape[1])
    z_hr = np.linspace(-1, 1, t1_shape[2])
    # For full inference we might need batching, but for 128^3 it's okayish?
    # T1 is likely 256^3 or similar, might run into RAM issues.
    # Let's verify shape.
    print(f"T1 Shape: {t1_shape}")
    
    # Batch the training
    batch_size = 10000
    dataset = (coords_lr, values_lr)
    
    # Model
    model = CoordinateMLP(jax.random.PRNGKey(0))
    optim = optax.adam(1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        def loss_fn(m, x, y):
            pred = jax.vmap(m)(x)
            return jnp.mean((pred - y)**2)
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    # Train Loop (Short for demo)
    print("Training INR...")
    for i in range(100): # 100 iters
        # Random batch
        idx = np.random.choice(len(coords_lr), batch_size)
        bx, by = coords_lr[idx], values_lr[idx]
        loss_val, model, opt_state = make_step(model, opt_state, bx, by)
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss_val}")
            
    # Inference
    print("Inferring HR...")
    # Process in chunks to save RAM
    out_vol = np.zeros(t1_shape)
    
    # Define a jitted inference function
    @eqx.filter_jit
    def predict_chunk(model, coords):
        return jax.vmap(model)(coords)

    # Very naive iteration over z-slices to save memory
    Z = t1_shape[2]
    # Creating meshgrid slice by slice
    xv_hr, yv_hr = np.meshgrid(x_hr, y_hr, indexing='ij')
    
    for z_idx in range(Z):
        zv = np.full_like(xv_hr, z_hr[z_idx])
        coords_slice = np.stack([xv_hr, yv_hr, zv], axis=-1).reshape(-1, 3)
        res = predict_chunk(model, coords_slice)
        out_vol[..., z_idx] = res.reshape(t1_shape[0], t1_shape[1])
        
    # Save
    out_dir = f"/home/mhough/datasets/ds001957-study/derivatives/super_resolution/sub-01/dwi"
    os.makedirs(out_dir, exist_ok=True)
    out_nii = nib.Nifti1Image(out_vol, t1_img.affine)
    nib.save(out_nii, os.path.join(out_dir, "sub-01_desc-inr_dwi.nii.gz"))
    print(f"Saved INR result: {out_dir}/sub-01_desc-inr_dwi.nii.gz")

if __name__ == "__main__":
    run_inr()
