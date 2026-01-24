
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import h5py
import argparse
import time
import pickle
import numpy as np
from dmipy_jax.inference.mdn import MixtureDensityNetwork, sample_posterior

def load_test_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = f['signals'][:]
        Y = f['parameters'][:]
    return X, Y

def load_jax_model(model_path, in_features, out_features, width_size=256, depth=3, num_components=8):
    key = jr.PRNGKey(0)
    model = MixtureDensityNetwork(
        in_features=in_features,
        out_features=out_features,
        num_components=num_components,
        width_size=width_size,
        depth=depth,
        key=key
    )
    model = eqx.tree_deserialise_leaves(model_path, model)
    return model

def load_norm_stats(pkl_path):
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
    return stats

def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets)**2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="uGUIDE_test_data.h5")
    parser.add_argument("--jax_model", type=str, default="examples/sbi/global_mdn.eqx")
    parser.add_argument("--uguide_results", type=str, default=None, help="Path to uGUIDE predictions .npy file")
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading test data...")
    X_raw, Y_gt = load_test_data(args.test_data)
    n_samples = X_raw.shape[0]
    
    # 2. Evaluate JAX Native
    print(f"Evaluating JAX Native on {n_samples} samples...")
    # Load stats
    stats = load_norm_stats(args.jax_model + ".pkl")
    X_mean, X_std = stats['X_mean'], stats['X_std']
    Y_mean, Y_std = stats['Y_mean'], stats['Y_std']
    
    X = (X_raw - X_mean) / X_std
    X = jnp.array(X, dtype=jnp.float32)
    
    # Load Model
    out_features = Y_gt.shape[1]
    model = load_jax_model(args.jax_model, in_features=X.shape[1], out_features=out_features)
    
    # Inference Speed Test
    # Warmup
    _ = jax.vmap(model)(X[0:1])
    
    start_time = time.time()
    # Batch inference
    # Note: MDN outputs parameters of mixture using `model(x)`, or samples using `sample_posterior`.
    # For accuracy, we usually take the mean of the mixture or the mode.
    # Let's calculate the expected value (mean) of the GMM.
    # Mean = sum(pi_k * mu_k)
    
    @eqx.filter_jit
    def get_mean_predictions(model, x):
        logits_pi, mu, log_sigma = jax.vmap(model)(x)
        pi = jax.nn.softmax(logits_pi, axis=-1)
        # mu: (N, K, D), pi: (N, K)
        # Weighted sum
        # expand pi: (N, K, 1)
        pi_exp = jnp.expand_dims(pi, -1)
        mean_pred = jnp.sum(pi_exp * mu, axis=1)
        return mean_pred
        
    preds_norm = get_mean_predictions(model, X)
    preds_norm.block_until_ready() # Sync
    
    duration = time.time() - start_time
    jax_speed = n_samples / duration
    
    # Un-normalize predictions
    preds = preds_norm * Y_std + Y_mean
    
    jax_rmse = rmse(preds, Y_gt)
    
    print("-" * 40)
    print(f"JAX NATIVE RESULTS")
    print(f"Inference Speed: {jax_speed:.2f} samples/sec")
    print(f"RMSE (Overall):  {jax_rmse:.6f}")
    
    # Per-parameter RMSE?
    # print(f"RMSE per param: {np.sqrt(np.mean((preds - Y_gt)**2, axis=0))}")
    print("-" * 40)
    
    # 3. Evaluate uGUIDE (Track A)
    if args.uguide_results:
        print(f"Loading uGUIDE results from {args.uguide_results}...")
        uguide_preds = np.load(args.uguide_results)
        
        uguide_rmse = rmse(uguide_preds, Y_gt)
        # Speed? User provides? Or we assume offline measure.
        print(f"uGUIDE OFFICIAL RESULTS")
        # print(f"Inference Speed: ??? samples/sec") # Placeholder
        print(f"RMSE (Overall):  {uguide_rmse:.6f}")
        print("-" * 40)
        
        # Comparison
        print(f"Speedup (JAX / uGUIDE): ??? (Requires uGUIDE speed)")
        print(f"RMSE Ratio (JAX / uGUIDE): {jax_rmse / uguide_rmse:.4f}")
    else:
        print("No uGUIDE results provided. Run Track A instructions to generate 'uguide_predictions.npy'.")
        print("See examples/sbi/uGUIDE_track_instructions.md")

if __name__ == "__main__":
    main()
