
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np

# Local Imports
from dmipy_jax.biophysics.neural_cde import NeuralCDE, GaussianPhaseApproximation

def generate_random_waveforms(n_samples, t_points, key):
    """
    Generates random smooth gradient waveforms (sum of sines).
    Shape: (N, T, 3)
    """
    dt = 1.0 / t_points
    times = jnp.linspace(0, 1, t_points)
    
    # Random Fourier Coefficients
    # G(t) = Sum( A_k sin(2 pi k t + phi_k) )
    n_freqs = 5
    
    # Batch keys
    keys = jax.random.split(key, n_samples)
    
    def single_waveform(k):
        # 3 Dimensions
        k_xyz = jax.random.split(k, 3)
        
        def axis_wave(k_ax):
            amps = jax.random.normal(k_ax, (n_freqs,))
            phases = jax.random.uniform(k_ax, (n_freqs,), minval=0, maxval=2*jnp.pi)
            
            # Construct wave
            wave = jnp.zeros_like(times)
            for i in range(n_freqs):
                freq = i + 1
                wave += amps[i] * jnp.sin(2 * jnp.pi * freq * times + phases[i])
            return wave
            
        gx = axis_wave(k_xyz[0])
        gy = axis_wave(k_xyz[1])
        gz = axis_wave(k_xyz[2])
        return jnp.stack([gx, gy, gz], axis=-1)
        
    return jax.vmap(single_waveform)(keys)

def generate_stejskal_tanner(t_points):
    """
    Standard PGSE Pulse (Trapezoid/Block).
    """
    times = jnp.linspace(0, 1, t_points)
    # Block pulse: 0.1 to 0.4, 0.6 to 0.9 (Refocusing implied or just two lobes)
    # Stejskal Tanner: +G for delta, -G for delta (if 180 included) or just effective G(t).
    # Let's model effective G(t) after 180: +G, then -G?
    # Or just +G, wait, +G (if bipolar).
    # Simple PGSE: +G (0.1-0.4), -G (0.6-0.9).
    
    g = jnp.zeros((t_points, 3))
    
    # X-gradient only
    mask1 = (times > 0.1) & (times < 0.4)
    mask2 = (times > 0.6) & (times < 0.9)
    
    # We set Gx = 1.0 (arbitrary high value)
    gx = jnp.where(mask1, 1.0, 0.0) + jnp.where(mask2, -1.0, 0.0)
    
    return jnp.stack([gx, jnp.zeros_like(gx), jnp.zeros_like(gx)], axis=-1)

def verify_neural_cde():
    print("=== Neural CDE Verification ===")
    
    # Config
    N_TRAIN = 50
    T_POINTS = 50
    DT = 1.0 / T_POINTS
    TRUE_D = 1e-3 # Diffusivity
    
    key = jax.random.PRNGKey(55)
    key_data, key_model = jax.random.split(key)
    
    # 1. Generate Data
    print("Generating training data (Random/Rough gradients)...")
    train_gradients = generate_random_waveforms(N_TRAIN, T_POINTS, key_data)
    times = jnp.linspace(0, 1, T_POINTS)
    
    # Compute Targets using Physics (GPA)
    # Target S = exp(-b D)
    # vmap the GPA
    def physics_forward(g):
        return GaussianPhaseApproximation.forward(g, DT, TRUE_D, gamma=10.0) # High gamma to induce contrast
        
    train_signals = jax.vmap(physics_forward)(train_gradients)
    # output (N, 1) needed
    train_signals = train_signals.reshape(-1, 1)
    
    print(f"Train Signal Mean: {jnp.mean(train_signals):.3f}")
    
    # 2. Setup CDE
    model = NeuralCDE(hidden_dim=16, key=key_model)
    optimizer = optax.adam(0.005)
    
    # Filter for learnable parameters (inexact arrays)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    
    # 3. Train Loop
    @eqx.filter_value_and_grad
    def loss_fn(m, x, y):
        # x: Gradients (T, 3)
        # y: Signal (1,)
        # Vmap over batch
        preds = jax.vmap(m, in_axes=(None, 0))(times, x)
        return jnp.mean((preds - y)**2)
        
    print("\nStarting Training...")
    for i in range(200):
        # We differentiate 'model', which is valid because eqx.filter_value_and_grad handles it.
        # But we must update only params.
        loss, grads = loss_fn(model, train_gradients, train_signals)
        
        # Get current params for update
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss = {loss:.6f}")
            
    # 4. Generalization Test
    print("\nTesting Generalization on Standard PGSE (Unseen waveform)...")
    pgse_wave = generate_stejskal_tanner(T_POINTS)
    true_pgse = physics_forward(pgse_wave).reshape(1)
    
    # Prediction
    pred_pgse = model(times, pgse_wave)
    
    err = float(jnp.squeeze((pred_pgse - true_pgse)**2))
    print(f"True PGSE Signal: {float(jnp.squeeze(true_pgse)):.4f}")
    print(f"Pred PGSE Signal: {float(jnp.squeeze(pred_pgse)):.4f}")
    print(f"Generalization Error (MSE): {err:.6f}")
    
    if err < 0.01:
        print("\nSUCCESS: Neural CDE generalized to unseen PGSE waveform.")
    else:
        print("\nWARNING: Generalization error high.")

if __name__ == "__main__":
    verify_neural_cde()
