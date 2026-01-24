
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import h5py
import argparse
import time
import pickle
from dmipy_jax.inference.mdn import MixtureDensityNetwork, mdn_loss

def load_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # X: Signals (inputs)
        X = f['signals'][:]
        # Y: Parameters (targets)
        Y = f['parameters'][:]
    
    # Normalize inputs? uGUIDE usually normalizes inputs/outputs.
    # Simple z-score normalization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0) + 1e-8
    
    return X, Y, (X_mean, X_std), (Y_mean, Y_std)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="uGUIDE_train_data.h5")
    parser.add_argument("--output_model", type=str, default="examples/sbi/global_mdn.eqx")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X_raw, Y_raw, (X_mean, X_std), (Y_mean, Y_std) = load_data(args.data)
    
    X = (X_raw - X_mean) / X_std
    Y = (Y_raw - Y_mean) / Y_std
    
    # Ensure float32
    X = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(Y, dtype=jnp.float32)
    
    n_samples, in_features = X.shape
    out_features = Y.shape[1]
    
    # Model
    print("Initializing model...")
    key = jr.PRNGKey(0)
    model = MixtureDensityNetwork(
        in_features=in_features,
        out_features=out_features,
        num_components=8, # Typical for uGUIDE is higher, maybe 10? 8 is reasonable.
        width_size=256,
        depth=3,
        key=key
    )
    
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        def batch_loss(m, x_b, y_b):
            return jnp.mean(jax.vmap(mdn_loss, in_axes=(None, 0, 0))(m, x_b, y_b))
            
        loss_val, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val
        
    # Stats
    # Save normalization stats to separate file or along with model
    norm_stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std
    }
    
    with open(args.output_model + ".pkl", 'wb') as f:
        pickle.dump(norm_stats, f)
        
    # Loop
    print(f"Training on {n_samples} samples for {args.epochs} epochs...")
    print(f"Input Features: {in_features}, Output Features: {out_features}")
    
    start_time = time.time()
    for epoch in range(args.epochs):
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n_samples)
        
        epoch_loss = 0.0
        steps = 0
        
        # JAX usually needs pre-split batches or scanning, but python loop is fine for this size
        for i in range(0, n_samples, args.batch_size):
            idx = perm[i:i+args.batch_size]
            batch_x = X[idx]
            batch_y = Y[idx]
            
            model, opt_state, loss = make_step(model, opt_state, batch_x, batch_y)
            epoch_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/steps:.4f}")
        
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    
    # Save model
    eqx.tree_serialise_leaves(args.output_model, model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()
