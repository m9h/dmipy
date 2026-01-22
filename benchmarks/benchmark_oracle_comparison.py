import jax
import jax.numpy as jnp
import numpy as np
import time
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.signal_models.cylinder_models import C1Stick

def benchmark_dmipy_jax():
    print("Starting Dmipy-Jax Benchmark (1M samples)...")
    
    # Enable JIT and Float32 (default)
    # jax.config.update("jax_enable_x64", False)
    
    # ==========================================
    # 1. Define Acquisition (Same as Oracle)
    # ==========================================
    n_dirs = 100
    np.random.seed(42)
    
    def random_directions(n):
        v = np.random.randn(n, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v
        
    bvecs_1000 = random_directions(n_dirs)
    bvals_1000 = np.ones(n_dirs) * 1000.0

    bvecs_2000 = random_directions(n_dirs)
    bvals_2000 = np.ones(n_dirs) * 2000.0

    bvals = np.concatenate([bvals_1000, bvals_2000])
    bvecs = np.concatenate([bvecs_1000, bvecs_2000])
    
    # Construct Acquisition Protocol
    # Note: Using standard units (s/mm^2) to match input numbers.
    acq = acquisition_scheme_from_bvalues(bvals, bvecs)
    
    # ==========================================
    # 2. Define Model (Ball & Stick)
    # ==========================================
    ball_comp = Ball()
    stick_comp = C1Stick()
    
    # Ball parameters: 'Ball_1_lambda_iso'
    # Stick parameters: 'C1Stick_1_lambda_par', 'C1Stick_1_mu'
    
    model = JaxMultiCompartmentModel([ball_comp, stick_comp])
    print(f"DEBUG: Model Parameter Names: {model.parameter_names}")
    
    # ==========================================
    # 3. Parameters (1M Samples)
    # ==========================================
    n_samples = 1_000_000
    
    # Diffusivity d = 1.7e-3
    d_val = 1.7e-3
    
    # Fraction f ~ U(0.1, 0.9)
    rng = np.random.default_rng(42)
    f_stick = rng.uniform(0.1, 0.9, n_samples)
    f_ball = 1.0 - f_stick
    
    # Orientation
    z = rng.uniform(-1, 1, n_samples)
    theta = np.arccos(z)
    phi = rng.uniform(0, 2*np.pi, n_samples)
    
    # mu (N, 2) for Stick (theta, phi)
    # C1Stick expects 'mu' as [theta, phi]
    mu = np.stack([theta, phi], axis=1)
    
    # Parameter Dictionary
    parameters = {}
    parameters['lambda_iso'] = jnp.full((n_samples,), d_val)
    parameters['lambda_par'] = jnp.full((n_samples,), d_val)
    parameters['mu'] = jnp.array(mu)
    
    # Partial Volumes
    parameters['partial_volume_0'] = jnp.array(f_ball) # Ball
    parameters['partial_volume_1'] = jnp.array(f_stick) # Stick
    
    # ==========================================
    # 4. Simulation & Timing
    # ==========================================
    
    print("JIT Compiling...")
    
    # We use vmap over the parameters to simulate a batch
    @jax.jit
    def simulate_batch(params):
        # vmap the model call. 
        # model(acq, **p) returns (G,) for single p.
        # vmapped returns (N, G).
        return jax.vmap(lambda p: model(p, acq))(params)

    # Warmup
    # Take a slice of parameters for warmup
    # Note: slicing dictionary of arrays
    p_warmup = {k: v[:100] for k,v in parameters.items()} # 100 samples warmup
    _ = simulate_batch(p_warmup).block_until_ready()
    
    print("Running simulation...")
    start_time = time.time()
    
    signals = simulate_batch(parameters).block_until_ready()
    
    # Add Rician Noise
    snr = 30.0
    sigma = 1.0 / snr
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    noise1 = jax.random.normal(k1, signals.shape) * sigma
    noise2 = jax.random.normal(k2, signals.shape) * sigma
    
    signals_noisy = jnp.sqrt((signals + noise1)**2 + noise2**2)
    signals_noisy.block_until_ready()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Simulation complete in {duration:.4f} seconds.")
    print(f"Throughput: {n_samples / duration:,.0f} samples/sec")
    
    # ==========================================
    # 5. Save
    # ==========================================
    out_file = "data/dmipy_jax_1M.npz"
    print(f"Saving to {out_file}...")
    np.savez(out_file, 
             signals=np.array(signals_noisy),
             signals_clean=np.array(signals),
             time_seconds=duration)
    
    print("Done!")

if __name__ == "__main__":
    benchmark_dmipy_jax()
