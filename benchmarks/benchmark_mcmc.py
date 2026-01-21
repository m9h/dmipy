import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import corner
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.models.ball_stick import BallStick
from dmipy_jax.inference.mcmc import MCMCInference
import time

def benchmark_mcmc_ball_stick():
    print("Setting up Benchmark...")
    
    # 1. Acquisition
    bvals = jnp.array([0] * 5 + [1000] * 30 + [2000] * 30)
    # Generate random bvecs on sphere
    rng = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(rng, (65, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    acquisition = JaxAcquisition(bvals, bvecs)
    
    # 3. Custom Model (Flexible BallStick)
    # Params: [f_intra, d_intra, d_iso, theta, phi]
    from dmipy_jax.cylinder import C1Stick
    from dmipy_jax.gaussian import G1Ball
    
    stick = C1Stick()
    ball = G1Ball()
    
    def flexible_model(params, acquisition):
        f_intra = params[0]
        d_intra = params[1]
        d_iso = params[2]
        theta = params[3]
        phi = params[4]
        
        mu = jnp.array([theta, phi])
        
        S_stick = stick(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=d_intra
        )
        
        S_ball = ball(
            bvals=acquisition.bvalues,
            lambda_iso=d_iso
        )
        
        return f_intra * S_stick + (1 - f_intra) * S_ball

    # Ground Truth Parameters
    # f_intra = 0.6, d_intra = 1.7e-9, d_iso = 3.0e-9 (using SI units usually, but let's stick to 1e-9 scale)
    # In dmipy, diffusivity is usually in m^2/s, so around 1e-9.
    # Previous code used 1.7e-3 which is massive if units are m^2/s. Assuming units are um^2/ms (1e-9 m^2/s).
    # IF acquisition bvals are in s/mm^2 (e.g. 1000), then D should be ~ 1e-3 mm^2/s.
    # Standard: b=1000 s/mm^2. D ~ 0.7e-3 to 3e-3 mm^2/s.
    
    true_params = jnp.array([0.6, 1.7e-3, 3.0e-3, jnp.pi/2, 0.0])
    
    # Generate signal
    clean_signal = flexible_model(true_params, acquisition)
    
    # Add noise
    sigma = 0.05
    rng_noise = jax.random.PRNGKey(1)
    noise = jax.random.normal(rng_noise, clean_signal.shape) * sigma
    data = jnp.abs(clean_signal + noise)
    
    print("Data generated. Starting MCMC...")
    start_time = time.time()
    
    # MCMC Setup
    # We fit all 5 parameters?
    # Constraints are tricky in unconstrained NUTS.
    # Blackjax NUTS works in unconstrained space usually.
    # If parameters are constrained (e.g. 0<f<1), we typically transform them.
    # For this basic benchmark, we might ignore constraints or use a wrapper with sigmoid.
    
    # Let's define a transformed model for unconstrained sampling
    def unconstrained_model(u_params, acq):
        # f_intra: sigmoid(u[0])
        f_intra = jax.nn.sigmoid(u_params[0])
        # d_intra: softplus(u[1]) * scale (e.g. 1e-3)
        d_intra = jax.nn.softplus(u_params[1]) * 1e-3
        # d_iso: softplus(u[2]) * scale
        d_iso = jax.nn.softplus(u_params[2]) * 1e-3
        # theta, phi: unconstrained? 
        # For full sphere, maybe treat as R^2? For now let's just pass through
        # But theta usually [0, pi], phi [-pi, pi].
        # Let's assume params are close enough or wrap them inside stick if needed.
        # Simple approach: just use raw values, stick usually handles periodicity or bounds?
        # Actually standard practice is to transform.
        # Let's keep it simple: just unconstrained for theta/phi.
        theta = u_params[3]
        phi = u_params[4]
        
        p = jnp.array([f_intra, d_intra, d_iso, theta, phi])
        return flexible_model(p, acq)
    
    # Initial guess (Inverse transform of reasonable values)
    # f=0.5 -> 0
    # d=1e-3 -> softplus_inv(1.0) ~ 1.0 (since softplus(1) ~ 1.3, softplus(0.5)~0.9)
    # let's start with u=1.0 for ds
    init_u_params = jnp.array([0.0, 1.0, 1.0, 1.57, 0.0])
    
    mcmc = MCMCInference(
        model_func=unconstrained_model,
        acquisition=acquisition,
        sigma=sigma,
        n_samples=1000,
        n_warmup=500
    )
    
    results = mcmc.fit(data, init_u_params, rng_key=jax.random.PRNGKey(42))
    samples_u = results['samples'] # Unconstrained samples
    
    duration = time.time() - start_time
    print(f"MCMC finished in {duration:.2f} seconds.")
    
    # Transform back to constrained space for plotting
    f_samples = jax.nn.sigmoid(samples_u[:, 0])
    d_intra_samples = jax.nn.softplus(samples_u[:, 1]) * 1e-3
    d_iso_samples = jax.nn.softplus(samples_u[:, 2]) * 1e-3
    theta_samples = samples_u[:, 3]
    phi_samples = samples_u[:, 4]
    
    samples_constrained = jnp.stack([f_samples, d_intra_samples, d_iso_samples, theta_samples, phi_samples], axis=1)
    
    # Plotting
    labels = [r"$f_{intra}$", r"$D_{intra}$", r"$D_{iso}$", r"$\theta$", r"$\phi$"]
    truths = [0.6, 1.7e-3, 3.0e-3, 1.57, 0.0]
    
    fig = corner.corner(
        np.array(samples_constrained), 
        labels=labels, 
        truths=truths,
        show_titles=True,
        title_fmt=".3f"
    )
    fig.savefig("benchmark_mcmc_corner.png")
    print("Saved benchmark_mcmc_corner.png")

if __name__ == "__main__":
    import numpy as np # corner needs numpy
    benchmark_mcmc_ball_stick()
