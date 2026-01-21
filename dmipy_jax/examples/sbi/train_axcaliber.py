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

from dmipy_jax.signal_models import cylinder_models
from dmipy_jax.signal_models.gaussian_models import g2_zeppelin

# --- 1. Simulation Logic ---

@jax.jit
def generate_protocol(n_shells=4, dirs_per_shell=16):
    """
    Generates an ActiveAx-like protocol:
    High Gradients up to 300 mT/m.
    Multiple Delta/delta combinations.
    """
    # ActiveAx Protocol Concept:
    # 4 shells of varying G and Delta.
    # G_max = 300 mT/m.
    # Big Delta (ms): [20, 30, 40, 50] ? 
    # Or keep standard Delta and vary G widely.
    # Let's target specific G/Delta combinations maximize sensitivity to ~1-5um.
    
    # Let's define 4 specific shells.
    # We return Arrays of shape (N_total,)
    
    # Shell 1: Low G, Long Delta (Standard-ish)
    # G=60 mT/m, Delta=40ms, delta=10ms => b ~ 1500
    
    # Shell 2: Mid G, Mid Delta
    # G=140 mT/m
    
    # Shell 3: High G, Short Delta
    # G=220 mT/m
    
    # Shell 4: Ultra High G, Short Delta
    # G=300 mT/m
    
    # In SI: 300 mT/m = 0.3 T/m.
    # Gyro = 2.675e8 rad/s/T.
    # q = gamma * G * delta / (2pi).
    # b = (gamma G delta)^2 (Delta - delta/3).
    
    # Let's set distinct parameters per shell.
    # N total = 4 * dirs_per_shell
    
    # Shell params (G_mT, Delta_ms, delta_ms)
    shell_params = jnp.array([
        [60., 40., 15.],
        [140., 30., 10.],
        [220., 20., 8.],
        [300., 15., 6.]
    ])
    
    # Convert to SI:
    # G: mT/m -> T/m (x 1e-3) -> * GAMMA -> rad/s/m
    GAMMA = 2.675152547e8
    G_si = shell_params[:, 0] * 1e-3
    Delta_si = shell_params[:, 1] * 1e-3
    delta_si = shell_params[:, 2] * 1e-3
    
    # Calc b-values
    # b = (gamma * G * delta)^2 * (Delta - delta/3)
    bvals_shells = (GAMMA * G_si * delta_si)**2 * (Delta_si - delta_si/3.0)
    # S/m^2. Convert to s/mm^2 for display? No, keep SI usually or match dmipy convention.
    # Cylinder model in dmipy_jax usually expects SI or consistent units?
    # dmipy defaults to SI (m, s). 
    # But usually b=1000 is s/mm^2 => 1e9 s/m^2.
    
    # Let's verify b-values order of magnitude.
    # Case 4: G=0.3, d=0.006, D=0.015.
    # q ~ 2.67e8 * 0.3 * 0.006 / 2pi ~ 76500 m^-1 => 76.5 mm^-1.
    # b ~ (2pi q)^2 * (D-d/3) ~ (4.8e5)^2 * (0.013) ~ 2.3e11 * 0.013 ~ 3e9 s/m^2 = 3000 s/mm^2.
    # Reasonable for High G axcaliber.
    
    # Construct full arrays
    bvals = []
    big_deltas = []
    small_deltas = []
    
    for i in range(n_shells):
        bvals.append(jnp.full(dirs_per_shell, bvals_shells[i]))
        big_deltas.append(jnp.full(dirs_per_shell, Delta_si[i]))
        small_deltas.append(jnp.full(dirs_per_shell, delta_si[i]))
        
    bvals = jnp.concatenate(bvals)
    big_deltas = jnp.concatenate(big_deltas)
    small_deltas = jnp.concatenate(small_deltas)
    
    # Directions: Random uniform on sphere per shell
    key = jax.random.PRNGKey(42)
    # Total dirs
    n_total = n_shells * dirs_per_shell
    # Reuse sample_on_sphere logic?
    z = jax.random.normal(key, (n_total, 3))
    bvecs = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    return bvals, bvecs, big_deltas, small_deltas

from functools import partial

# Global Protocol
BVALS, BVECS, BIG_DELTAS, SMALL_DELTAS = generate_protocol()
PROTOCOL_SIZE = len(BVALS)

@partial(jax.jit, static_argnames=['batch_size'])
def get_batch(key, batch_size=128):
    """
    Simulates AxCaliber signal: Intracellular Cylinder + Extracellular Zeppelin.
    """
    k_params, k_noise = jax.random.split(key)
    
    # 1. Sample Parameters
    # Diameter: 0.1 - 5.0 um -> 0.1e-6 - 5.0e-6 m
    d_min, d_max = 0.1e-6, 5.0e-6
    # Log-uniform sampling for diameter to handle orders of magnitude better? 
    # Or uniform? "Priors: Axon radii ... Gamma" implies distribution of radii?
    # Prompt says: "Prior for Axon Radii ... Gamma distribution parameters IF modeling a distribution."
    # But later: "Train NPE to predict Mean Axon Diameter (MAD)" and "Sensitivity vs Ground Truth Radius".
    # This implies we simulate single-radius voxels (ActiveAx style) or we simulate a distribution and predict its mean.
    # Single-radius is cleaner for the validation plot requested.
    # Let's sample Diameter Uniformly for now to cover the range well in training data.
    
    diameters = jax.random.uniform(k_params, (batch_size,), minval=d_min, maxval=d_max)
    
    # Volume Fraction (f_intra): 0.1 - 0.9
    f_intra = jax.random.uniform(k_params, (batch_size,), minval=0.1, maxval=0.9)
    # Make f_intra (N,) -> (N, 1) for broadcasting
    f_intra = f_intra[:, None]
    
    # Orientation: Random
    # We assume parallel fibers (Cylinder || Zeppelin).
    # Sample random mu on sphere.
    # To use in model, we need (theta, phi) or cartesian.
    # dmipy_jax Cylinder model takes cartesian if passed as mu? 
    # Let's check `RestrictedCylinder.__call__`: "if mu_cart.ndim > 1..."
    # It converts scalar theta/phi to cartesian.
    # But let's see if we pass cartesian directly?
    # implementation: `mu = jnp.asarray(mu); if mu.ndim > 0: theta=mu[0]...`
    # It seems to EXPECT spherical coords [theta, phi] in the implementation of RestrictedCylinder.__init__ but __call__ handles conversion.
    # Wait, `c2_cylinder` takes `mu` (cartesian). 
    # `RestrictedCylinder` converts param to cart.
    # Let's construct a vmapped functional call that bypasses the class overhead for speed, 
    # like in `train_dti.py`, OR update `RestrictedCylinder` to handle batched inputs nicely.
    # For now, functional approach: `cylinder_models.c2_cylinder`
    
    k_dir, k_sub = jax.random.split(k_params)
    z = jax.random.normal(k_dir, (batch_size, 3))
    mu_cart = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    # Diffusivities
    # lambda_par: ~ 1.7 um^2/ms = 1.7e-9 m^2/s.
    # Let's vary slightly: 1.5 - 2.0
    lambda_par = jax.random.uniform(k_sub, (batch_size,), minval=1.5e-9, maxval=2.2e-9)
    
    # Extra-cellular perp diffusivity (lambda_perp for Zeppelin)
    # usually < lambda_par. Say 0.5 - 1.0 e-9
    lambda_perp = jax.random.uniform(k_sub, (batch_size,), minval=0.5e-9, maxval=1.2e-9)
    
    # 2. Simulate Signal
    
    # Signal = f * S_cyl + (1-f) * S_zep
    
    # Intra: Restricted Cylinder
    # c2_cylinder(bvals, bvecs, mu, lambda_par, diameter, big_delta, small_delta)
    # We need to vmap this over the batch.
    # PROTOCOL (BVALS et al) is fixed for all samples (broadcasted).
    
    def simulate_single_pixel(d, f, mu, l_par, l_perp):
        # Broadcast protocol against single pixel params
        # bvals: (M,)
        # mu: (3,)
        
        # S_cyl
        s_cyl = cylinder_models.c2_cylinder(
            BVALS, BVECS, mu, l_par, d, BIG_DELTAS, SMALL_DELTAS
        )
        
        # S_zep
        # Zeppelin: c2_zeppelin(bvals, bvecs, mu, lambda_par, lambda_perp)
        # Check zeppelin signature.
        s_zep = g2_zeppelin(
            BVALS, BVECS, mu, l_par, l_perp
        )
        
        return f * s_cyl + (1.0 - f) * s_zep

    signals = jax.vmap(simulate_single_pixel)(diameters, f_intra.ravel(), mu_cart, lambda_par, lambda_perp)
    
    # 3. Add Noise
    # SNR ~ 30-50 for high quality data.
    # But High G data is noisy. Let's say SNR=30 at b=0.
    sigma = 1.0 / 30.0
    k_n1, k_n2 = jax.random.split(k_noise, 2)
    n1 = jax.random.normal(k_n1, signals.shape) * sigma
    n2 = jax.random.normal(k_n2, signals.shape) * sigma
    signals_noisy = jnp.sqrt((signals + n1)**2 + n2**2)
    
    # 4. Targets: Diameter
    # We want to predict Diameter.
    # Scale Diameter to um for easier training (0.1 - 5.0)
    targets = diameters * 1e6 
    # Shape (N,) -> (N, 1)
    targets = targets[:, None]
    
    return signals_noisy, targets

# --- 2. Inference Network (MDN) ---
# Copied from train_dti.py for standalone usage
class MixtureDensityNetwork(eqx.Module):
    full_shared_mlp: eqx.nn.MLP
    n_components: int
    n_outputs: int
    
    def __init__(self, key, in_size, out_size, n_components=8, width=128, depth=4):
        self.n_components = n_components
        self.n_outputs = out_size
        
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
        raw_out = self.full_shared_mlp(x)
        nc = self.n_components
        no = self.n_outputs
        
        logits = raw_out[:nc]
        means = raw_out[nc : nc + nc*no].reshape(nc, no)
        log_sigmas = raw_out[nc + nc*no:].reshape(nc, no)
        sigmas = jnp.exp(log_sigmas)
        
        return logits, means, sigmas

def mdn_loss_fn(model, x, y):
    logits, means, sigmas = model(x)
    y_b = y
    z = (y_b - means) / sigmas
    log_prob_comps = -0.5 * jnp.sum(z**2, axis=-1) - jnp.sum(jnp.log(sigmas), axis=-1) - 0.5 * means.shape[1] * jnp.log(2*jnp.pi)
    log_pis = jax.nn.log_softmax(logits)
    final_log_prob = jax.scipy.special.logsumexp(log_pis + log_prob_comps)
    return -final_log_prob

def loss_fn(model, x_batch, y_batch):
    per_sample_loss = jax.vmap(mdn_loss_fn, in_axes=(None, 0, 0))(model, x_batch, y_batch)
    return jnp.mean(per_sample_loss)

# --- 3. Training Loop ---

def main():
    print(f"Running AxCaliber SBI training on device: {jax.devices()[0]}")
    print(f"Protocol: {PROTOCOL_SIZE} measurements.")
    print(f"Max Gradient: {300} mT/m")
    
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 256
    N_ITERATIONS = 500
    LOG_INTERVAL = 100
    
    key = jax.random.PRNGKey(42)
    key_net, key_data = jax.random.split(key)
    
    # Input: Signals (PROTOCOL_SIZE). Output: Diameter (1)
    model = MixtureDensityNetwork(
        key_net, 
        in_size=PROTOCOL_SIZE, 
        out_size=1,
        n_components=6,
        width=256,
        depth=5
    )
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

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

    # --- 4. Validation & Sensitivity Plot ---
    print("Generating sensitivity plot...")
    
    # Generate Test Data spanning the diameter range densely
    n_test = 2000
    key_iter, k_test = jax.random.split(key_iter)
    
    # We want to see performance vs Diameter specifically.
    # Let's manually Linspace the diameters to ensure coverage
    d_test_vals = jnp.linspace(0.1e-6, 5.0e-6, n_test)
    # Fix other params or random? Random is better to marginalize over them.
    f_test = jax.random.uniform(k_test, (n_test,), minval=0.1, maxval=0.9)[:, None]
    
    k_rot, k_diff = jax.random.split(k_test)
    z = jax.random.normal(k_rot, (n_test, 3))
    mu_test = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    
    lp_test = jax.random.uniform(k_diff, (n_test,), minval=1.5e-9, maxval=2.2e-9)
    lperp_test = jax.random.uniform(k_diff, (n_test,), minval=0.5e-9, maxval=1.2e-9)
    
    def simulate_test(d, f, mu, l_par, l_perp):
        s_cyl = cylinder_models.c2_cylinder(BVALS, BVECS, mu, l_par, d, BIG_DELTAS, SMALL_DELTAS)
        s_zep = g2_zeppelin(BVALS, BVECS, mu, l_par, l_perp)
        return f * s_cyl + (1.0 - f) * s_zep
        
    signals_clean = jax.vmap(simulate_test)(d_test_vals, f_test.ravel(), mu_test, lp_test, lperp_test)
    
    # Add noise
    sig = 1.0/30.0
    n1 = jax.random.normal(k_test, signals_clean.shape) * sig # Reuse key ok for viz script
    n2 = jax.random.normal(k_rot, signals_clean.shape) * sig
    x_test = jnp.sqrt((signals_clean + n1)**2 + n2**2)
    y_test = d_test_vals * 1e6 # um
    
    # Get Posterior predictions
    # We want mean and StdDev of the posterior.
    # E[y] = sum pi * mu
    # Var[y] = sum pi * (sigma^2 + mu^2) - (E[y])^2
    
    @eqx.filter_jit
    def predict_stats(model, x):
        logits, means, sigmas = model(x)
        weights = jax.nn.softmax(logits) # (K,)
        # means (K, 1), sigmas (K, 1) usually
        
        mean_pred = jnp.sum(weights[:, None] * means, axis=0)
        
        # Second moment
        second_mom = jnp.sum(weights[:, None] * (sigmas**2 + means**2), axis=0)
        var_pred = second_mom - mean_pred**2
        std_pred = jnp.sqrt(jnp.clip(var_pred, 1e-9, None))
        
        return mean_pred, std_pred

    mus, stds = jax.vmap(predict_stats, in_axes=(None, 0))(model, x_test)
    # mus, stds: (N, 1)
    
    mus = mus.ravel()
    stds = stds.ravel()
    
    # PLOT
    plt.figure(figsize=(8, 6))
    
    # Scatter of Mean Prediction
    plt.scatter(y_test, mus, c='b', s=2, alpha=0.3, label='Predictions')
    plt.plot([0, 5], [0, 5], 'k--', lw=1, label='Identity')
    
    # Plot Error Bars / Confidence Interval
    # To visualize "spread" properly:
    # Plot spread vs GT
    
    plt.fill_between(y_test, mus - stds, mus + stds, color='gray', alpha=0.3, label='Posterior StdDev')
    
    plt.xlabel('Ground Truth Diameter (um)')
    plt.ylabel('Predicted Diameter (um)')
    plt.title('AxCaliber Inference (High G=300mT/m)')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(os.path.dirname(__file__), 'axcaliber_sensitivity.png')
    plt.savefig(out_path)
    print(f"Sensitivity plot saved to {out_path}")
    
    # Check Theoretical Limit
    # For small diameters (< 2um), do we see higher uncertainty or bias?
    # Ideally, below a certain limit, the prediction flattens to the prior mean, and uncertainty becomes prior width.
    # Prior was 0.1-5.0 uniform. Mean=2.55. Std ~ 1.4.
    
    # Let's inspect mean std for small vs large
    mask_small = y_test < 2.0
    mask_large = y_test > 3.0
    
    std_small = jnp.mean(stds[mask_small])
    std_large = jnp.mean(stds[mask_large])
    
    print(f"Mean Uncertainty (StdDev) for R < 2um: {std_small:.4f}")
    print(f"Mean Uncertainty (StdDev) for R > 3um: {std_large:.4f}")
    
    if std_small > std_large:
        print("SUCCESS: Uncertainty is higher for small axons, reflecting the resolution limit.")
    else:
        print("WARNING: Uncertainty pattern is unexpected. Check model or noise.")

if __name__ == "__main__":
    main()
