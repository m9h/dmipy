
import numpy as np
import h5py
import pickle
import torch
import argparse
import time
from pathlib import Path

# uGUIDE imports
from uGUIDE.config_utils import create_config_uGUIDE, save_config_uGUIDE, load_config_uGUIDE
from uGUIDE.data_utils import preprocess_data
from uGUIDE.inference import run_inference
from uGUIDE.estimation import estimate_microstructure

def load_data(h5_file):
    with h5py.File(h5_file, 'r') as f:
        X = f['signals'][:]
        Y = f['parameters'][:]
    return X, Y

def get_bvals(n_b0=1, n_b1000=64, n_b2500=64):
    bvals = np.concatenate([
        np.zeros(n_b0),
        np.ones(n_b1000) * 1000,
        np.ones(n_b2500) * 2500
    ])
    return bvals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="uGUIDE_train_data.h5")
    parser.add_argument("--test_data", type=str, default="uGUIDE_test_data.h5")
    parser.add_argument("--output_dir", type=str, default="examples/sbi/uguide_model")
    parser.add_argument("--epochs", type=int, default=5) # Kept low for benchmark demo, usually 50+
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    X_train, theta_train = load_data(args.train_data)
    X_test, theta_test_gt = load_data(args.test_data)
    
    # Recreate bvals used in generation (1 b0, 64 b1000, 64 b2500)
    # Note: If generation changed, this needs to match.
    # In generate_synthetic_training_data.py: 1 b0, 64 b1000, 64 b2500.
    bvals = get_bvals()
    
    # 2. Preprocess (Normalize)
    print("Preprocessing data...")
    # uGUIDE preprocess expects numpy
    theta_train, X_train = preprocess_data(theta_train, X_train, bvals, normalize=True)
    theta_test_gt, X_test = preprocess_data(theta_test_gt, X_test, bvals, normalize=True)
    
    # 3. Config
    # Define priors. Since we don't strictly enforce the prior range in uGUIDE inference (it learns from data),
    # we can set the ranges to min/max of data or standard ranges.
    # We have 13 parameters. Let's name them generic p0-p12.
    prior = {}
    for i in range(theta_train.shape[1]):
        p_min = theta_train[:, i].min()
        p_max = theta_train[:, i].max()
        # Add a little buffer
        margin = (p_max - p_min) * 0.1
        prior[f'p{i}'] = [p_min - margin, p_max + margin]
        
    config = create_config_uGUIDE(
        microstructure_model_name='Generic_3Compartment',
        size_x=X_train.shape[1],
        prior=prior,
        prior_postprocessing=None,
        use_MLP=True, # Use MLP like GlobalMDN for fair comparison
        nf_features=6, # Irrelevant if MLP?
        max_epochs=args.epochs,
        n_epochs_no_change=5,
        nb_samples=100, # Number of samples for posterior estimation during inference?
        random_seed=1234,
        folderpath=Path(args.output_dir)
    )
    
    save_config_uGUIDE(config, savefile='config.pkl')
    print(f"Device: {config['device']}")
    
    # 4. Train
    config_path = Path(args.output_dir) / 'config.pkl'
    if config_path.exists():
        print(f"Model found at {config_path}. Skipping training.")
        config = load_config_uGUIDE(config_path)
    else:
        print("Starting training...")
        start_train = time.time()
        run_inference(theta_train, X_train, config=config, plot_loss=False, load_state=False)
        train_time = time.time() - start_train
        print(f"Training finished in {train_time:.2f}s")
    
    # 5. Inference on Test Set
    print("Running inference on test set...")
    from uGUIDE.estimation import sample_posterior_distribution
    
    start_inf = time.time()
    
    predictions = []
    
    for i in range(len(X_test)):
        # x input shape: (N_features,)
        samples = sample_posterior_distribution(X_test[i], config)
        # samples shape: (nb_samples, n_params)
        mean_est = samples.mean(axis=0)
        predictions.append(mean_est)
        
        if i % 100 == 0:
            print(f"Inference {i}/{len(X_test)}")
            
    end_inf = time.time()
    inf_speed = len(X_test) / (end_inf - start_inf)
    print(f"Inference Speed: {inf_speed:.2f} samples/sec")
    
    predictions = np.array(predictions)
    
    # Save results
    np.save('uguide_predictions.npy', predictions)
    print("Saved uguide_predictions.npy")

if __name__ == "__main__":
    main()
