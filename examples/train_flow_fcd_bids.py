import os
import json
import time
import jax

jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import nibabel as nib
from tqdm import tqdm

from dmipy_jax.models.flow_fcd import FlowUNet
from dmipy_jax.fitting.flow_trainer import train_step_ot_cfm, generate_flair
from dmipy_jax.io.deepfcd_loader import DeepFCDLoader

def save_bids_derivative(output_dir: str, subject_id: str, flair_data: np.ndarray, meta: dict, affine=None):
    """Saves generated FLAIR in BIDS derivatives format."""
    subj_out_dir = os.path.join(output_dir, subject_id, 'anat')
    os.makedirs(subj_out_dir, exist_ok=True)
    
    # NIfTI
    fname_nii = os.path.join(subj_out_dir, f"{subject_id}_desc-generated_FLAIR.nii.gz")
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(flair_data, affine)
    nib.save(img, fname_nii)
    
    # JSON Sidecar
    fname_json = os.path.join(subj_out_dir, f"{subject_id}_desc-generated_FLAIR.json")
    with open(fname_json, 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    # Configuration
    DATA_PATH = os.path.expanduser("~/Data/ds001957")
    OUTPUT_DIR = os.path.join(DATA_PATH, "derivatives", "deepfcd-flow")
    BATCH_SIZE = 2
    LR = 1e-4
    EPOCHS = 1 # Demo run (Smoke Test)
    STEPS_PER_EPOCH = 2 
    IN_CHANNELS = 7
    
    # Ensure derivatives dataset_description
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "dataset_description.json"), 'w') as f:
        json.dump({
            "Name": "DeepFCD Flow Generation",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "dmipy-jax-flow",
                    "Version": "0.1.0",
                    "Description": "Conditional Flow Matching for FCD Detection"
                }
            ]
        }, f, indent=2)
    
    print(f"Initializing Training on {DATA_PATH}...")
    
    # Init Loader
    loader = DeepFCDLoader(DATA_PATH, slice_neighbors=3)
    train_gen = loader.get_generator(batch_size=BATCH_SIZE)
    
    # Init Model
    key = jax.random.PRNGKey(42)
    key_model, key_train = jax.random.split(key, 2)
    
    # Initialize Model (Reduced size for smoke test)
    # in_channels = 1 (State) + 7 (Context) = 8
    model = FlowUNet(in_channels=8, out_channels=1, base_dim=16, key=key_model)
    
    # Optimizer
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    
    losses = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        start_time = time.time()
        
        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for i in range(STEPS_PER_EPOCH):
                # Data
                context, target = next(train_gen)
                
                # JAX Array conversion
                context = jnp.array(context) # (B, 7, H, W)
                target = jnp.array(target)   # (B, 1, H, W)
                
                key_train, step_key = jax.random.split(key_train)
                
                model, opt_state, loss = train_step_ot_cfm(
                    model, opt_state, target, context, step_key, optimizer
                )
                
                loss_val = loss.item()
                epoch_loss += loss_val
                pbar.set_postfix({'loss': f"{loss_val:.4f}"})
                pbar.update(1)
                
        avg_loss = epoch_loss / STEPS_PER_EPOCH
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} done in {time.time()-start_time:.2f}s. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint (eqx.tree_serialise)
        # eqx.tree_serialise_leaves(os.path.join(OUTPUT_DIR, f"model_epoch_{epoch+1}.eqx"), model)
        
    print("Training Complete.")
    
    # Inference Demo on Real Subject (sub-01)
    print("Running Inference on sub-01...")
    sub01_t1, sub01_flair = loader.load_subject("sub-01")
    
    if sub01_t1 is not None:
        # Generate for a center slice
        z = sub01_t1.shape[2] // 2
        
        # Context: (7, H, W)
        ctx_vol = sub01_t1[:, :, z-3 : z+4]
        ctx_vol = np.transpose(ctx_vol, (2, 0, 1))
        ctx_jax = jnp.array(ctx_vol)[None, ...] # (1, 7, H, W) - wait, generate_flair expects unbatched context?
        # Check generate_flair signature:
        # def generate_flair(model, context, key): context is (C, H, W)
        
        ctx_jax_single = jnp.array(ctx_vol) # (7, H, W)
        
        key_gen = jax.random.PRNGKey(2026)
        gen_flair = generate_flair(model, ctx_jax_single, key_gen) # (1, H, W)
        
        # Save BIDS output
        # Re-orient to (H, W, 1) for saving as 3D-ish slice or just 2D
        gen_flair_np = np.array(gen_flair).transpose(1, 2, 0) # (H, W, 1)
        
        meta = {
            "Model": "OT-CFM FlowUNet",
            "Epochs": EPOCHS,
            "SigmaMin": 1e-4,
            "Solver": "Tsit5",
            "SliceIndex": z
        }
        
        save_bids_derivative(OUTPUT_DIR, "sub-01", gen_flair_np, meta)
        print(f"Saved inference output to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
