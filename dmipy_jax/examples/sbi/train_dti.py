import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import sys
import os

# Ensure project root is in path if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.signal_models import gaussian_models

# --- 1. Simulation Logic ---

def compute_fa_md(lambda_1, lambda_2, lambda_3):
    """Computes Fractional Anisotropy (FA) and Mean Diffusivity (MD)."""
    md = (lambda_1 + lambda_2 + lambda_3) / 3.0
    
    # FA calculation
    num = (lambda_1 - md)**2 + (lambda_2 - md)**2 + (lambda_3 - md)**2
    denom = lambda_1**2 + lambda_2**2 + lambda_3**2
    
    # numer/denom can be 0. Avoid NaN.
    # FA = sqrt(3/2 * num / denom)
    fa_sq = 1.5 * num / (denom + 1e-9)
    fa = jnp.sqrt(jnp.clip(fa_sq, 0.0, 1.0))
    
    return fa, md

def sample_on_sphere(key, shape):
    """Samples uniform random unit vectors on the sphere."""
    # Method: Normal distribution normalized
    z = jax.random.normal(key, shape + (3,))
    return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

def simulate_dti_data(key, n_samples=1000):
    """
    Generates synthetic DTI data.
    
    Returns:
        signals: (n_samples, n_bvals)
        targets: (n_samples, 2)  [FA, MD]
    """
    
    # --- Acquisition Scheme ---
    # Standard single-shell: b=1000, 32 dirs
    # For simplicity, we can fix these or generate them.
    # Let's fix 32 directions for training stability, or randomized for robustness?
    # Usually fixed scheme per scanner protocol. Let's sample random directions ONCE 
    # if we want a fixed protocol, or sample random b-vecs every batch for "amortized" over schemes.
    # Instruction says "Use a standard single-shell scheme".
    
    # Let's define a fixed set of directions for the scope of this simulation 
    # to mimic a specific acquisition execution.
    # Or better: random directions per sample might be too hard for MLP if it doesn't take gradients as input.
    # Standard NPE usually assumes fixed context (gradients). 
    # So we define fixed gradients.
    
    n_dirs = 32
    bval = 1000.0 # s/mm^2 = 1e6 s/m^2?
    # NOTE: biological ranges in prompt are 0.1 - 3.0 um^2/ms = 0.1e-9 - 3.0e-9 m^2/s.
    # If using SI units (dmipy defaults to SI usually):
    # b=1000 s/mm^2 = 1000 * 1e6 s/m^2 = 1e9 s/m^2.
    
    bvals = jnp.full(n_dirs, 1e9)
    
    # deterministic sub-key for fixed directions if we wanted, 
    # but here we are inside a jitted function likely, so let's just make them "random" 
    # but same for the batch? No, wait. 
    # If the network inputs ONLY signal, the gradient scheme MUST be fixed and known implicitly.
    # If we change gradients, we must input them.
    # Let's fix them roughly evenly distributed. 
    # For now, simple random fixed set for the whole training process is typical for single-protocol NPE.
    # We will generate them outside or inside with a fixed seed if we want consistency?
    # Let's generate random ones per call implies varying protocol. 
    # We will stick to a fixed protocol assumption for this simple MLP.
    
    # Hack: use a fixed seed for directions to ensure they are constant across batches?
    # Or pass them in.
    # To keep it simple and efficient in JAX:
    # We will assume the acquisition is FIXED. 
    # We generate "fixed" directions inside this function using a hardcoded key if needed, or 
    # we just pass them. Let's make them global constant for now.
    pass

import functools

# Fixed Acquisition Scheme (Global for simplicity of this script)
N_DIRS = 32
B_VALUE = 1e9 # 1000 s/mm^2 in SI
RS = jax.random.PRNGKey(42)
FIXED_BVECS = sample_on_sphere(RS, (N_DIRS,))
FIXED_BVALS = jnp.full((N_DIRS,), B_VALUE)

@functools.partial(jax.jit, static_argnames=['batch_size'])
def get_batch(key, batch_size=128):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    
    # 1. Sample Parameters
    # Eigenvalues: 0.1 - 3.0 um^2/ms = 0.1e-9 - 3.0e-9 m^2/s
    l_min, l_max = 0.1e-9, 3.0e-9
    l1 = jax.random.uniform(k1, (batch_size,), minval=l_min, maxval=l_max)
    l2 = jax.random.uniform(k2, (batch_size,), minval=l_min, maxval=l1) # l2 <= l1
    l3 = jax.random.uniform(k3, (batch_size,), minval=l_min, maxval=l2) # l3 <= l2
    
    # Orientations (Euler angles)
    # alpha, gamma: -pi to pi
    # beta: 0 to pi
    alpha = jax.random.uniform(k4, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    beta = jax.random.uniform(k5, (batch_size,), minval=0.0, maxval=jnp.pi)
    gamma = jax.random.uniform(k1, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    
    # 2. Simulate Signal
    # dmipy_jax Tensor model
    # We need to vectorize the call. 
    # The gaussian_models.Tensor class is an eqx.Module, but the functional `g2_tensor` is jitted.
    # Let's use the functional interface for speed in batch generation.
    
    # g2_tensor expects single inputs? No, JAX usually supports broadcasting if implemented right.
    # Looking at g2_tensor implementation:
    # It constructs e1, e2 from angles.
    # class Tensor.__call__ handles the angle conversion.
    
    # Let's instantiate a mapped model or just write the call manually to ensure batching.
    # The provided class `Tensor` seems to handle single point.
    # We can vmap the `Tensor.__call__` or the functional `g2_tensor` logic.
    
    # Let's use vmap on the class method logic.
    def forward_single(l1, l2, l3, a, b, g):
        model = gaussian_models.Tensor(lambda_1=l1, lambda_2=l2, lambda_3=l3, 
                                       alpha=a, beta=b, gamma=g)
        return model(FIXED_BVALS, FIXED_BVECS)
    
    signals = jax.vmap(forward_single)(l1, l2, l3, alpha, beta, gamma)
    
    # 3. Add Noise
    # Rician Noise: S_noisy = sqrt( (S + n1)^2 + n2^2 )
    # SNR = 30. Noise Sigma = S_b0 / SNR. S_b0 is approx 1.0 (theoretical).
    # So sigma = 1/30.
    sigma = 1.0 / 30.0
    k_n1, k_n2 = jax.random.split(k2, 2)
    n1 = jax.random.normal(k_n1, signals.shape) * sigma
    n2 = jax.random.normal(k_n2, signals.shape) * sigma
    
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)
    
    # 4. Compute Targets (FA, MD)
    fa, md = compute_fa_md(l1, l2, l3)
    
    # Normalize targets for easier training?
    # MD is 1e-9 scale. FA is 0-1.
    # Let's scale MD by 1e9 so it's approx 0.1-3.0.
    targets = jnp.stack([fa, md * 1e9], axis=-1)
    
    return signals_noisy, targets


# --- 2. Inference Network (MDN) ---

class MixtureDensityNetwork(eqx.Module):
    full_shared_mlp: eqx.nn.MLP
    n_components: int
    n_outputs: int
    
    def __init__(self, key, in_size, out_size, n_components=8, width=128, depth=3):
        self.n_components = n_components
        self.n_outputs = out_size
        
        # Output size:
        # Weights (logits): n_components
        # Means: n_components * out_size
        # Variances (log_sigma): n_components * out_size
        total_out = n_components * (1 + 2 * out_size)
        
        self.full_shared_mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=total_out,
            width_size=width,
            depth=depth,
            activation=jax.nn.gelu,
            key=key
        )
    
    def __call__(self, x):
        # x: (in_size,)
        raw_out = self.full_shared_mlp(x)
        
        # Split outputs
        nc = self.n_components
        no = self.n_outputs
        
        logits = raw_out[:nc] # (K,)
        means = raw_out[nc : nc + nc*no].reshape(nc, no) # (K, D)
        log_sigmas = raw_out[nc + nc*no:].reshape(nc, no) # (K, D)
        
        # Apply constraints
        # Weights: softmax via logits in loss usually, or here.
        # Sigmas: exp
        sigmas = jnp.exp(log_sigmas)
        
        return logits, means, sigmas

def mdn_loss_fn(model, x, y):
    """
    Negative Log Likelihood of the Mixture.
    y: (D,)
    """
    logits, means, sigmas = model(x) # (K,), (K, D), (K, D)
    
    # Log-likelihood of each component:
    # log N(y | mu, sigma) = -0.5 * ((y - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
    # Sum over dimensions D (assuming diagonal covariance)
    
    # y broadcasted to (K, D)
    y_b = y # (D,)
    
    z = (y_b - means) / sigmas # (K, D)
    log_prob_comps = -0.5 * jnp.sum(z**2, axis=-1) - jnp.sum(jnp.log(sigmas), axis=-1) - 0.5 * means.shape[1] * jnp.log(2*jnp.pi)
    # log_prob_comps: (K,)
    
    # Mixture log-likelihood:
    # log sum_k ( pi_k * exp(log_prob_k) )
    # = log_sum_exp( log_pi_k + log_prob_k )
    
    log_pis = jax.nn.log_softmax(logits) # (K,)
    
    final_log_prob = jax.scipy.special.logsumexp(log_pis + log_prob_comps)
    
    return -final_log_prob

# Batch loss
def loss_fn(model, x_batch, y_batch):
    # Vectorize over batch
    per_sample_loss = jax.vmap(mdn_loss_fn, in_axes=(None, 0, 0))(model, x_batch, y_batch)
    return jnp.mean(per_sample_loss)

# --- 3. Training Loop ---

def main():
    print(f"Running on device: {jax.devices()[0]}")
    
    # Hyperparams
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    N_ITERATIONS = 5000
    LOG_INTERVAL = 500
    
    # Initialize Key
    key = jax.random.PRNGKey(0)
    key_net, key_data = jax.random.split(key)
    
    # Initialize Model
    # Input size = 32 (signals)
    # Output size = 2 (FA, MD)
    model = MixtureDensityNetwork(
        key_net, 
        in_size=N_DIRS, 
        out_size=2,
        n_components=8,
        width=128,
        depth=3
    )
    
    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Update Step
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    # Training
    print("Starting training...")
    key_iter = key_data
    losses = []
    
    for i in range(N_ITERATIONS):
        key_iter, k_batch = jax.random.split(key_iter)
        x_batch, y_batch = get_batch(k_batch, batch_size=BATCH_SIZE)
        
        model, opt_state, loss = make_step(model, opt_state, x_batch, y_batch)
        losses.append(loss)
        
        if (i + 1) % LOG_INTERVAL == 0:
            print(f"Iter {i+1}/{N_ITERATIONS}, Loss: {loss:.4f}")
            
    print("Training complete.")
    
    # --- 4. Validation & Plotting ---
    print("Generating validation plot...")
    
    # Generate Test Set
    key_iter, k_test = jax.random.split(key_iter)
    x_test, y_test = get_batch(k_test, batch_size=1000)
    
    # Prediction (Max Mode or Mean?)
    # For a multimodal posterior, mean can be bad. Mode is better.
    # But for DTI parameters which are fairly unimodal usually, expected value is fine.
    # Let's compute Expected Value for simplicity of plotting.
    # E[y] = sum_k pi_k mu_k
    
    @eqx.filter_jit
    def predict_mean(model, x):
        logits, means, _ = model(x)
        weights = jax.nn.softmax(logits) # (K,)
        # sum weighted means
        # means (K, D)
        # weights (K, 1)
        w = weights[:, None]
        pred = jnp.sum(w * means, axis=0)
        return pred

    preds = jax.vmap(predict_mean, in_axes=(None, 0))(model, x_test)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # FA
    axes[0].scatter(y_test[:, 0], preds[:, 0], alpha=0.3, s=5)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel("True FA")
    axes[0].set_ylabel("Predicted FA")
    axes[0].set_title("Fractional Anisotropy")
    axes[0].grid(True)
    
    # MD (Original scale was 1e-9, we trained on 1e9 scaled)
    axes[1].scatter(y_test[:, 1], preds[:, 1], alpha=0.3, s=5)
    axes[1].plot([0, 3], [0, 3], 'k--', lw=1)
    axes[1].set_xlabel("True MD (10^-9 m^2/s)")
    axes[1].set_ylabel("Predicted MD")
    axes[1].set_title("Mean Diffusivity")
    axes[1].grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'dti_inference_results.png')
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
