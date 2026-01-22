
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np

# Local Imports
import sys
import os
sys.path.append(os.getcwd())
try:
    from dmipy_jax.biophysics.neural_cde import NeuralCDE
except ImportError:
    # Fallback to relative if running from root
    from dmipy_jax.biophysics.neural_cde import NeuralCDE

def generate_random_waveforms(n_samples, t_points, key):
    """Generates random smooth gradient waveforms with variable amplitude and dimensionality."""
    times = jnp.linspace(0, 1, t_points)
    n_freqs = 5
    keys = jax.random.split(key, n_samples)
    
    def single_waveform(k):
        k_amp, k_ax_mask, k_waves = jax.random.split(k, 3)
        
        # 1. Random Amplitude Scaling (Uniform [10, 80] to cover test case of 30)
        scale = jax.random.uniform(k_amp, minval=10.0, maxval=80.0)
        
        # 2. Random Axis Masking (Simulate 1D/2D/3D acquisitions)
        logits = jax.random.normal(k_ax_mask, (3,))
        mask = logits > -0.5 # Bias slightly towards active
        # Ensure at least one is True
        all_false = jnp.logical_not(jnp.any(mask))
        mask = mask.at[0].set(jnp.logical_or(mask[0], all_false))
        mask = mask.astype(jnp.float32)

        def axis_wave(k_ax):
            amps = jax.random.normal(k_ax, (n_freqs,))
            phases = jax.random.uniform(k_ax, (n_freqs,), minval=0, maxval=2*jnp.pi)
            wave = jnp.zeros_like(times)
            for i in range(n_freqs):
                freq = (i + 1) * 2 
                wave += amps[i] * jnp.sin(2 * jnp.pi * freq * times + phases[i])
            return wave 
            
        k_xyz = jax.random.split(k_waves, 3)
        gx = axis_wave(k_xyz[0]) * mask[0]
        gy = axis_wave(k_xyz[1]) * mask[1]
        gz = axis_wave(k_xyz[2]) * mask[2]
        
        raw_wave = jnp.stack([gx, gy, gz], axis=-1)
        return raw_wave * scale # Apply scale
        
    return jax.vmap(single_waveform)(keys)

def generate_sine_ogse(t_points, freq):
    """Generates a Sine-OGSE waveform (Sine modulated gradient)."""
    times = jnp.linspace(0, 1, t_points)
    # G(t) = sin(2 pi f t) envelope
    # Simple sine for whole duration
    gx = jnp.sin(2 * jnp.pi * freq * times)
    # Effective gradient
    return jnp.stack([gx, jnp.zeros_like(gx), jnp.zeros_like(gx)], axis=-1) * 30.0 # High amplitude

def restricted_diffusion_spectral_ground_truth(gradients, dt, D0=2.0e-3, Gamma=50.0):
    """
    Simulates restricted diffusion using the Gaussian Phase Approximation (Frequency Domain).
    Spectrum D(omega) is modeled as a Lorentzian: D(w) = D0 * Gamma^2 / (w^2 + Gamma^2).
    """
    # 1. Compute q(t)
    gamma_gyro = 1.0 # Normalized
    qt = jnp.cumsum(gradients, axis=0) * dt * gamma_gyro
    
    # 2. FFT to get Q(omega)
    N = gradients.shape[0]
    # Q_w = fft(qt) * dt
    Q_w = jnp.fft.fft(qt, axis=0) * dt
    
    # Frequencies 
    freqs = jnp.fft.fftfreq(N, d=dt) # Hz
    omegas = 2 * jnp.pi * freqs
    
    # 3. Spectrum D(omega)
    # Lorentzian
    # D_w = D0 * Gamma^2 / (omega^2 + Gamma^2)
    D_w = D0 * (Gamma**2) / (omegas**2 + Gamma**2)
    
    # 4. Integrate
    # Power Spectrum |Q(w)|^2
    df = 1.0 / (N * dt)
    # Q_w is (N, 3). Sum over xyz.
    power_Q = jnp.sum(jnp.abs(Q_w)**2, axis=-1)
    
    # Attenuation exponent
    exponent = jnp.sum( D_w * power_Q ) * df
    
    return jnp.exp(-exponent)

def verify_complex():
    print("=== Neural CDE Verification (Restricted/Spectral Model) ===")
    
    # Config
    N_TRAIN = 500
    T_POINTS = 100
    DT = 1.0 / T_POINTS
    
    key = jax.random.PRNGKey(42)
    key_data, key_model = jax.random.split(key)
    
    # 1. Generate Data
    print("Generating training data (Random variable waveforms)...")
    train_gradients = generate_random_waveforms(N_TRAIN, T_POINTS, key_data)
    times = jnp.linspace(0, 1, T_POINTS)
    
    # Compute Targets using Spectral GPA
    print("Computing Ground Truth (Lorentzian Spectrum)...")
    def physics_forward(g):
        val = restricted_diffusion_spectral_ground_truth(g, DT)
        return jnp.real(val) # Enforce Real
        
    train_signals = jax.vmap(physics_forward)(train_gradients)
    train_signals = train_signals.reshape(-1, 1)
    
    print(f"Train Signal Mean: {jnp.mean(train_signals):.3f}")
    
    # 2. Setup CDE
    model = NeuralCDE(hidden_dim=32, key=key_model)
    optimizer = optax.adam(0.002)
    
    # 3. Train Loop
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss_fn(m, x, y):
        preds = jax.vmap(m, in_axes=(None, 0))(times, x)
        return jnp.mean((preds - y)**2)
        
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = loss_fn(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("\nStarting Training (1000 Epochs)...")
    for i in range(1000):
        loss, model, opt_state = make_step(model, opt_state, train_gradients, train_signals)
        
        if i % 100 == 0:
            print(f"Iter {i}: Loss = {loss:.6f}")
            
    # 4. Generalization Test
    print("\nTesting Generalization on Sine-OGSE (Unseen structured waveform)...")
    # Low freq OGSE
    ogse_low = generate_sine_ogse(T_POINTS, freq=3.0)
    true_low = jnp.real(restricted_diffusion_spectral_ground_truth(ogse_low, DT))
    pred_low = model(times, ogse_low)
    
    # High freq OGSE
    ogse_high = generate_sine_ogse(T_POINTS, freq=10.0)
    true_high = jnp.real(restricted_diffusion_spectral_ground_truth(ogse_high, DT))
    pred_high = model(times, ogse_high)
    
    print("--- Results ---")
    print(f"Low Freq (3Hz)  | True: {float(jnp.squeeze(true_low)):.4f} | Pred: {float(jnp.squeeze(pred_low)):.4f}")
    print(f"High Freq (10Hz)| True: {float(jnp.squeeze(true_high)):.4f} | Pred: {float(jnp.squeeze(pred_high)):.4f}")
    
    err_low = float(jnp.squeeze((pred_low - true_low)**2))
    err_high = float(jnp.squeeze((pred_high - true_high)**2))
    print(f"MSE Low: {err_low:.6f}")
    print(f"MSE High: {err_high:.6f}")
    
    if (err_low + err_high)/2 < 0.02:
        print("\nSUCCESS: Neural CDE captured frequency-dependence (spectrum).")
    else:
        print("\nWARNING: Generalization error high.")

if __name__ == "__main__":
    verify_complex()
