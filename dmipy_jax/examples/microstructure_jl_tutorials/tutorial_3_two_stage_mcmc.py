
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dmipy_jax.inference.mcmc import MCMCInference
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.gaussian import G1Ball

def main():
    print("Tutorial 3: Two-Stage MCMC")
    
    # 1. Setup Model (NODDI-like: Stick + Ball)
    # Simple model to fit: Stick (intra) + Ball (extra)
    # Signals = f * S_stick + (1-f) * S_ball
    # Params: theta, phi, f
    # Fixed: lambda_par, lambda_iso
    
    stick = C1Stick()
    ball = G1Ball()
    
    # Constants
    lambda_par = 1.7e-9
    lambda_iso = 3.0e-9
    
    def model_func(params, acquisition):
        theta, phi, f = params
        mu = jnp.array([theta, phi])
        
        # Constrain f to (0,1) via sigmoid? MCMC handles unconstrained usually, 
        # or we assume params are in unconstrained space.
        # Microstructure.jl uses bijectors usually.
        # But our simple MCMC example in dmipy_jax seems to assume parameters are in the space, 
        # or we need to add transforms. 
        # For simplicity here, let's assume f is unconstrained but we clip or sigmoid it inside if needed.
        # Or just trust the prior/likelihood to keep it reasonable? 
        # Blackjax NUTS usually works on unconstrained space R^n.
        # Let's add simple sigmoid transform for f.
        
        real_f = jax.nn.sigmoid(f)
        
        S_stick = stick(acquisition.bvalues, acquisition.gradient_directions, 
                       mu=mu, lambda_par=lambda_par)
        S_ball = ball(acquisition.bvalues, lambda_iso=lambda_iso)
        
        return real_f * S_stick + (1 - real_f) * S_ball

    # 2. Synthetic Data
    # True params
    true_theta = 0.5
    true_phi = 1.0
    true_f_logit = 0.0 # f=0.5
    true_params = jnp.array([true_theta, true_phi, true_f_logit])
    
    # Acquisition
    bvals = jnp.repeat(jnp.array([1000.0, 2000.0]) * 1e6, 30)
    key = jax.random.PRNGKey(42)
    bvecs = jax.random.normal(key, (60, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # Generate noisy signal
    clean_signal = model_func(true_params, acq)
    noise_sigma = 0.05
    noisy_signal = clean_signal + noise_sigma * jax.random.normal(key, clean_signal.shape)
    
    # Stage 1: Quick Initialization (e.g. Grid Search or just a heuristic guess)
    # In real pipeline, we'd run AMICO or NLS.
    # Here let's just perturb the true params to simulate a rough starting point.
    init_params = true_params + jnp.array([0.2, -0.2, 0.5]) 
    print("Initial Guess:", init_params)
    
    # Stage 2: MCMC
    # Initialize MCMC Inference
    mcmc = MCMCInference(model_func, acq, sigma=noise_sigma, n_samples=2000, n_warmup=1000)
    
    print("Running MCMC Chains...")
    rng_key = jax.random.PRNGKey(101)
    
    # data, initial_params
    result = mcmc.fit(noisy_signal, init_params, rng_key)
    samples = result['samples']
    
    print("MCMC Completed. Sample shape:", samples.shape)
    
    # In unconstrained space
    f_samples_logit = samples[:, 2]
    f_samples = jax.nn.sigmoid(f_samples_logit)
    
    theta_samples = samples[:, 0]
    phi_samples = samples[:, 1]
    
    print(f"Post. Mean f: {jnp.mean(f_samples):.3f} (True: 0.5)")
    print(f"Post. Mean theta: {jnp.mean(theta_samples):.3f} (True: {true_theta})")
    
    # Plot Corner plot (Trace plots)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    axes[0].plot(theta_samples)
    axes[0].axhline(true_theta, color='r')
    axes[0].set_ylabel('Theta')
    
    axes[1].plot(phi_samples)
    axes[1].axhline(true_phi, color='r')
    axes[1].set_ylabel('Phi')
    
    axes[2].plot(f_samples)
    axes[2].axhline(0.5, color='r')
    axes[2].set_ylabel('Fraction f')
    
    plt.suptitle('Tutorial 3: MCMC Trace Plots')
    plt.savefig('tutorial_3_output.png')
    print("Saved MCMC trace plots to tutorial_3_output.png")

if __name__ == "__main__":
    main()
