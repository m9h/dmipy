
import jax
import jax.numpy as jnp
import equinox as eqx
import time
import numpy as np
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.inference.amortized import ZeppelinNetwork
from dmipy_jax.fitting.optimization import OptimistixFitter

def main():
    print("Benchmarking Amortized Inference vs NLLS Oracle (Zeppelin)...")
    
    # 1. Setup
    key = jax.random.PRNGKey(42)
    n_voxels = 100 
    n_measurements = 64
    
    # Acquisition (Multi-shell for Zeppelin stability)
    bvals = jnp.concatenate([jnp.zeros(4), jnp.ones(30)*1000, jnp.ones(30)*2000])
    vecs = jax.random.normal(key, (n_measurements, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=vecs)
    
    # 2. Ground Truth Generation
    # Parameters: [lambda_par, lambda_perp, theta, phi]
    # Ranges: 
    # lambda_par: [0.5, 3.0] um^2/ms * 1e-9
    # lambda_perp: [0.1, 1.0] um^2/ms * 1e-9
    key, subkey = jax.random.split(key)
    lambda_par_gt = jax.random.uniform(subkey, (n_voxels,), minval=0.5e-9, maxval=3.0e-9)
    lambda_perp_gt = jax.random.uniform(subkey, (n_voxels,), minval=0.1e-9, maxval=1.0e-9)
    
    # Ensure par > perp
    lambda_par_gt = jnp.maximum(lambda_par_gt, lambda_perp_gt + 0.2e-9)
    
    theta_gt = jax.random.uniform(subkey, (n_voxels,), minval=0, maxval=jnp.pi)
    phi_gt = jax.random.uniform(subkey, (n_voxels,), minval=0, maxval=2*jnp.pi)
    
    # Convert angles to mu (cartesian)
    st = jnp.sin(theta_gt)
    ct = jnp.cos(theta_gt)
    sp = jnp.sin(phi_gt)
    cp = jnp.cos(phi_gt)
    mu_gt = jnp.stack([st*cp, st*sp, ct], axis=1) # (N, 3)
    
    # Generate Signal
    # Using vmap to generate signal for all voxels
    def gen_signal(lp, lperp, m):
        z = Zeppelin(mu=m, lambda_par=lp, lambda_perp=lperp)
        return z(bvals, vecs)
        
    signals = jax.vmap(gen_signal)(lambda_par_gt, lambda_perp_gt, mu_gt)
    
    # Add noise
    signals = signals + 0.02 * jax.random.normal(key, signals.shape)
    
    print(f"Dataset: {n_voxels} voxels, {n_measurements} measurements.")

    # ---------------------------------------------------------
    # 3. Amortized Inference Benchmark
    # ---------------------------------------------------------
    print("\n--- Amortized Inference (Forward Pass) ---")
    network = ZeppelinNetwork(key, n_measurements)
    
    # Vmap the network
    inference_fn = eqx.filter_vmap(network)
    jit_inference = eqx.filter_jit(inference_fn)
    
    # Warmup
    _ = jit_inference(signals[:10])
    
    start_time = time.time()
    preds_dict = jit_inference(signals)
    jax.block_until_ready(preds_dict['lambda_par']) # Sync
    end_time = time.time()
    
    amortized_time = end_time - start_time
    print(f"Time: {amortized_time:.4f} s")
    print(f"Throughput: {n_voxels / amortized_time:.2f} voxels/s")
    
    # Accuracy Check
    # Extract predicted arrays
    lpar_pred = preds_dict['lambda_par']
    
    # MSE
    mse_lpar = jnp.mean((lpar_pred - lambda_par_gt)**2)
    print(f"MSE lambda_par: {mse_lpar:.2e} (untrained network)")

    # ---------------------------------------------------------
    # 4. Oracle Benchmark (NLLS - Optimistix LM)
    # ---------------------------------------------------------
    print("\n--- Oracle NLLS (Optimistix LM) ---")
    
    # Define wrapper for Fitter
    # Zeppelin Wrapper: params are [lambda_par, lambda_perp, theta, phi]
    def zeppelin_wrapper(params, acq):
        lp, lperp, th, ph = params
        # Optimistix is unconstrained, so we might get negative values.
        # We should use softplus/sigmoid internally or just let it fly for simple benchmark.
        # Using abs() for safety on physical params
        lp = jnp.abs(lp)
        lperp = jnp.abs(lperp)
        
        # Reconstruct mu
        st = jnp.sin(th)
        ct = jnp.cos(th)
        sp = jnp.sin(ph)
        cp = jnp.cos(ph)
        mu = jnp.array([st*cp, st*sp, ct])
        
        z = Zeppelin(mu=mu, lambda_par=lp, lambda_perp=lperp)
        return z(acq.bvalues, acq.gradient_directions)

    # Ranges for scaling only (bounds ignored by basic LM)
    # lp, lperp, th, ph
    ranges = [
        (0.1e-9, 5.0e-9), # lambda_par
        (0.01e-9, 3.0e-9), # lambda_perp
        (0.0, jnp.pi),     # theta
        (0.0, 2*jnp.pi)    # phi
    ]
    
    scales = [1e-9, 1e-9, 1.0, 1.0]
    
    fitter = OptimistixFitter(zeppelin_wrapper, ranges, scales=scales)
    
    # Initial Guess
    init_params = jnp.array([1.5e-9, 0.5e-9, 1.0, 1.0])
    
    # Vmap fit
    vfit = jax.vmap(fitter.fit, in_axes=(0, None, 0))
    jit_fit = jax.jit(vfit)
    
    # Prepare inputs
    all_init = jnp.tile(init_params, (n_voxels, 1))
    
    # Warmup
    _ = jit_fit(signals[:10], acq, all_init[:10])
    
    start_time = time.time()
    fitted_params, match_result = jit_fit(signals, acq, all_init)
    jax.block_until_ready(fitted_params)
    end_time = time.time()
    
    nlls_time = end_time - start_time
    print(f"Time: {nlls_time:.4f} s")
    print(f"Throughput: {n_voxels / nlls_time:.2f} voxels/s")
    
    # Accuracy Check
    # Need to handle potential neg values from unconstrained fit
    lpar_nlls = jnp.abs(fitted_params[:, 0]) 
    mse_lpar_nlls = jnp.mean((lpar_nlls - lambda_par_gt)**2)
    print(f"MSE lambda_par: {mse_lpar_nlls:.2e}")
    
    # Comparisons
    speedup = nlls_time / amortized_time
    print(f"\nSpeedup (Amortized / NLLS): {speedup:.1f}x")


if __name__ == "__main__":
    main()
