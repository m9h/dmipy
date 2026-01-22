
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
import argparse
import time
import os
import glob
from dmipy_jax.nn.fwe import FreeWaterNet

def main():
    parser = argparse.ArgumentParser(description="Train FreeWaterNet Oracle")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (file or dir)")
    parser.add_argument("--model_out", type=str, default="models/fwe_oracle.eqx", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--finetune_from", default=None, help="Path to pretrained model for fine-tuning")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.data}...")
    if os.path.isdir(args.data):
        # Load all .npz in directory
        files = glob.glob(os.path.join(args.data, "*.npz"))
        print(f"Found {len(files)} files.")
        all_signals = []
        all_targets = []
        for f in files:
            d = np.load(f)
            all_signals.append(d['signals'])
            # Handle key difference: synthetic uses 'f_iso', real uses 'f_iso_gt'
            if 'f_iso' in d:
                all_targets.append(d['f_iso'])
            elif 'f_iso_gt' in d:
                all_targets.append(d['f_iso_gt'])
            else:
                raise KeyError(f"No f_iso or f_iso_gt in {f}")
        signals = jnp.concatenate(all_signals, axis=0) # (Total, n_dirs)
        targets = jnp.concatenate(all_targets, axis=0)
    else:
        # Single file
        data = np.load(args.data)
        signals = jnp.array(data['signals']) # (N, n_dirs)
        if 'f_iso' in data:
            targets = jnp.array(data['f_iso'])
        elif 'f_iso_gt' in data:
            targets = jnp.array(data['f_iso_gt'])
        else:
            raise KeyError("No f_iso or f_iso_gt in data")
    
    # Flatten if 4D (Volumetric Real Data)
    if signals.ndim > 2:
        print(f"Flattening input with shape {signals.shape}...")
        signals = signals.reshape(-1, signals.shape[-1])
        targets = targets.reshape(-1)
        
    n_samples, n_inputs = signals.shape
    
    # Split Train/Val (80/20)
    n_train = int(0.8 * n_samples)
    train_x, val_x = signals[:n_train], signals[n_train:]
    train_y, val_y = targets[:n_train], targets[n_train:]
    
    print(f"Train samples: {n_train}, Val samples: {n_samples - n_train}")
    
    # 2. Initialize Model
    key = jax.random.PRNGKey(0)
    key_init, key_train = jax.random.split(key)
    
    model = FreeWaterNet(in_size=n_inputs, key=key_init)
    
    if args.finetune_from:
        print(f"Fine-tuning from {args.finetune_from}...")
        model = eqx.tree_deserialise_leaves(args.finetune_from, model)
    
    # 3. Optimization Setup
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Loss Function
    def loss_fn(model, x, y):
        # x: (batch, n_inputs)
        # y: (batch,)
        pred = jax.vmap(model)(x) # (batch,)
        return jnp.mean((pred - y)**2)
    
    # Update Step
    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Validation Step
    @eqx.filter_jit
    def evaluate(model, x, y):
        return loss_fn(model, x, y)

    # 4. Training Loop
    n_batches = n_train // args.batch_size
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Shuffle
        key_train, k_shuffle = jax.random.split(key_train)
        perm = jax.random.permutation(k_shuffle, n_train)
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]
        
        epoch_loss = 0.0
        
        for i in range(n_batches):
            batch_x = train_x_shuffled[i*args.batch_size : (i+1)*args.batch_size]
            batch_y = train_y_shuffled[i*args.batch_size : (i+1)*args.batch_size]
            
            model, opt_state, loss = step(model, opt_state, batch_x, batch_y)
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / n_batches
        
        # Validation
        val_loss = evaluate(model, val_x, val_y).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    print(f"Final Validation MSE: {val_loss:.6f}")
    
    # 5. Save Model
    eqx.tree_serialise_leaves(args.model_out, model)
    print(f"Model saved to {args.model_out}")

if __name__ == "__main__":
    main()
