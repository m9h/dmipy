import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from dmipy_jax.models.flow_fcd import FlowUNet
from dmipy_jax.fitting.flow_trainer import train_step_ot_cfm, generate_flair
from dmipy_jax.io.deepfcd_loader import simulated_loader

def main():
    print("Initializing OT-CFM verification...")
    
    # Hyperparams
    BATCH_SIZE = 8
    LR = 1e-4
    STEPS = 100
    IN_CHANNELS = 7 # T1 context 2.5D stack
    OUT_CHANNELS = 1 # FLAIR
    
    key = jax.random.PRNGKey(42)
    key_model, key_train = jax.random.split(key)
    
    # Model [Channels: In=1 (FLAIR) + 7 (Context)] = Vector Field Input Dimension?
    # Wait, FlowUNet definition:
    # __call__(x, t, context) -> concatenates x (1) + context (7).
    # Init: in_channels should be 1. 
    # But FlowUNet `start_conv` takes (in_channels + context_channels)?
    # Let's check FlowUNet.__call__:
    # inp = jnp.concatenate([x, context], axis=0)
    # start_conv(inp)
    # So start_conv in_channels must be 1 + 7 = 8.
    
    model = FlowUNet(in_channels=8, out_channels=1, key=key_model)
    
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Data
    loader = simulated_loader(BATCH_SIZE, in_channels=IN_CHANNELS)
    
    # Training Loop
    losses = []
    print(f"Training for {STEPS} steps...")
    
    for i, (context, target) in tqdm(zip(range(STEPS), loader), total=STEPS):
        # context: (B, 7, H, W)
        # target: (B, 1, H, W) -> x_1
        
        key_train, step_key = jax.random.split(key_train)
        
        model, opt_state, loss = train_step_ot_cfm(
            model, opt_state, target, context, step_key, optimizer
        )
        losses.append(loss)
        
        if i % 20 == 0:
            print(f"Step {i}: Loss = {loss:.4f}")
            
    # Inference Verification
    print("Running inference on one sample...")
    ctx_sample = context[0] # (7, H, W)
    target_sample = target[0] # (1, H, W)
    
    gen_flair = generate_flair(model, ctx_sample, key_train) # (1, H, W)
    
    print(f"Generated shape: {gen_flair.shape}")
    print(f"Target shape: {target_sample.shape}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Context (Center Slice)")
    plt.imshow(ctx_sample[3], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Generated FLAIR")
    plt.imshow(gen_flair[0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Real FLAIR")
    plt.imshow(target_sample[0], cmap='gray')
    plt.axis('off')
    
    plt.savefig("verification_flow.png")
    print("Verification visualization saved to verification_flow.png")

if __name__ == "__main__":
    main()
