
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from dmipy_jax.inference.mcmc import MCMCInference
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.distributions.sphere_distributions import SD1Watson
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
from dmipy_jax.cylinder import C1Stick

# Enable x64 for precision if needed, but float32 usually fine for MCMC
# jax.config.update("jax_enable_x64", True)

def load_protocol(file_path):
    """
    Parses MDT protocol file.
    Format: #gx,gy,gz,Delta,delta,TE,b,TR
    """
    try:
        data = np.loadtxt(file_path, skiprows=1) # Skip header
    except Exception as e:
        print(f"Error loading protocol: {e}")
        # Fallback or strict error
        raise e
        
    g_dirs = data[:, 0:3]
    # Normalize g_dirs
    norms = np.linalg.norm(g_dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    g_dirs = g_dirs / norms
    
    b_values = data[:, 6]
    # Check units. Paper uses SI in simulations usually? 
    # The file has 1.000000e+09 which is 1000 s/mm^2 in SI (s/m^2).
    # Normal DWI b=1000 is 1e9 s/m^2.
    # So these are SI units.
    
    delta = data[:, 4]
    Delta = data[:, 3]
    TE = data[:, 5]
    
    return JaxAcquisition(
        bvalues=jnp.array(b_values),
        gradient_directions=jnp.array(g_dirs),
        delta=jnp.array(delta),
        Delta=jnp.array(Delta),
        echo_time=jnp.array(TE)
    )

def build_noddi_model():
    # Intra: Stick + Watson
    stick = C1Stick()
    watson_ic = SD1Watson(grid_size=50) # Reduced Grid size for speed
    intra = DistributedModel(stick, watson_ic, target_parameter='mu')
    
    # Extra: Zeppelin + Watson
    zeppelin = G2Zeppelin()
    watson_ec = SD1Watson(grid_size=50)
    extra = DistributedModel(zeppelin, watson_ec, target_parameter='mu')
    
    # Iso: Ball
    ball = G1Ball()
    
    # Combined
    model = JaxMultiCompartmentModel([intra, extra, ball])
    return model

def reproduction_workflow():
    print("----------------------------------------------------------------")
    print("   Reproduction of Uncertainty Paper (MCMC Validation)          ")
    print("   Using dmipy-jax                                              ")
    print("----------------------------------------------------------------")
    
    # 1. Load Protocol
    protocol_path = "benchmarks/external/uncertainty_paper/data/simulations/hcp_mgh_1003.prtcl"
    if not os.path.exists(protocol_path):
        print(f"Protocol file not found at {protocol_path}")
        return
        
    acq = load_protocol(protocol_path)
    print(f"Protocol Loaded: {len(acq.bvalues)} measurements.")
    
    # 2. Setup Model
    noddi = build_noddi_model()
    # Check param names for debugging
    # print("Model params:", noddi.parameter_names) 
    
    # 3. Ground Truth Data Generation
    # Params: S0=1e4, fic_global=0.5, fiso=0.2, kappa=10
    # Derived: fec_global=0.3. 
    # vic (intra-volume-fraction in tissue) = fic / (fic+fec) = 0.5/0.8 = 0.625
    # d_par = 1.7e-9, d_iso=3.0e-9
    # d_perp = 1.7e-9 * (1 - 0.625)
    
    S0_true = 10000.0
    fic_global_true = 0.5
    fiso_true = 0.2
    kappa_true = 10.0
    theta_true = np.pi / 2.0
    phi_true = np.pi / 2.0
    
    d_par_fixed = 1.7e-9
    d_iso_fixed = 3.0e-9
    
    # Fractions for MCM (weights)
    w_ic = fic_global_true
    w_ec = 0.3 # 1 - 0.5 - 0.2
    w_iso = fiso_true
    
    # Derived parameters
    v_ic = w_ic / (w_ic + w_ec)
    d_perp = d_par_fixed * (1.0 - v_ic)
    
    mu_true = jnp.array([theta_true, phi_true])[None, :] # Shape (1, 2)
    
    # Intra (0): mu, kappa, lambda_par
    # Extra (1): mu_2, kappa_2, lambda_par_2, lambda_perp
    # Ball (2): lambda_iso
    # Fractions: partial_volume_0, partial_volume_1, partial_volume_2
    
    gt_params = {
        'mu': mu_true,
        'kappa': jnp.array([kappa_true]),
        'lambda_par': jnp.array([d_par_fixed]),
        
        'mu_2': mu_true, # Coupled
        'kappa_2': jnp.array([kappa_true]), # Coupled
        'lambda_par_2': jnp.array([d_par_fixed]),
        'lambda_perp': jnp.array([d_perp]),
        
        'lambda_iso': jnp.array([d_iso_fixed]),
        
        'partial_volume_0': jnp.array([w_ic]),
        'partial_volume_1': jnp.array([w_ec]),
        'partial_volume_2': jnp.array([w_iso])
    }
    
    # Simulate
    E_clean = noddi(gt_params, acq)
    S_clean = S0_true * E_clean
    
    # Add Noise (SNR=30)
    SNR = 30.0
    sigma = S0_true / SNR
    key = jax.random.PRNGKey(42)
    k1, k2, k_mcmc = jax.random.split(key, 3)
    
    n1 = jax.random.normal(k1, S_clean.shape) * sigma
    n2 = jax.random.normal(k2, S_clean.shape) * sigma
    S_noisy = jnp.sqrt((S_clean + n1)**2 + n2**2)
    
    print(f"Simulated Data with SNR={SNR}. Signal shape: {S_noisy.shape}")
    
    # 4. MCMC Inference
    # We need to define a model function that maps Sampling Parameters -> Model Params (dict)
    # Sampling Params:
    # 0: f_iso (logit)
    # 1: v_ic (logit) -> defines f_ic, f_ec and d_perp
    # 2: kappa (log)
    # 3: theta
    # 4: phi
    # 5: S0 (log?) or fixed? Usually S0 is nuisance or we give it a prior.
    # Let's simple sample S0 too.
    
    def model_func(params_array, acquisition):
        # Unpack
        f_iso_logit = params_array[0]
        v_ic_logit = params_array[1]
        kappa_log = params_array[2]
        theta = params_array[3]
        phi = params_array[4]
        S0 = params_array[5] 
        
        # Transform
        f_iso = jax.nn.sigmoid(f_iso_logit)
        v_ic = jax.nn.sigmoid(v_ic_logit)
        kappa = jnp.exp(kappa_log)
        
        # Derived
        w_ic = (1.0 - f_iso) * v_ic
        w_ec = (1.0 - f_iso) * (1.0 - v_ic)
        w_iso = f_iso
        
        d_perp_ec = d_par_fixed * (1.0 - v_ic)
        
        mu_vec = jnp.array([theta, phi])[None, :]
        
        # Build Dict
        p = {
            'mu': mu_vec,
            'kappa': kappa,
            'lambda_par': d_par_fixed,
            
            'mu_2': mu_vec,
            'kappa_2': kappa,
            'lambda_par_2': d_par_fixed,
            'lambda_perp': d_perp_ec,
            
            'lambda_iso': d_iso_fixed,
            
            'partial_volume_0': w_ic,
            'partial_volume_1': w_ec,
            'partial_volume_2': w_iso
        } # Ensure batch dims if needed, but MCMC usually vmaps calls so scalar is fine if code supports it.
        # But wait, my manual mu_vec has (1,2).
        # MCMCInference might pass scalar params?
        # MCMCInference vmaps the model_func over CHAINS.
        # Inside one call, params are single sample.
        # So mu_vec shape (1,2) is correct for JaxMultiCompartmentModel if it expects batch dim.
        
        return S0 * noddi(p, acquisition)

    # Initial Guess (Perturbed GT)
    # 0: fiso=0.2 -> logit(-1.38)
    # 1: vic=0.625 -> logit(0.51)
    # 2: kappa=10 -> log(2.3)
    # 3: theta=1.57
    # 4: phi=1.57
    # 5: S0=10000
    
    init_params = jnp.array([
        jax.scipy.special.logit(0.2),
        jax.scipy.special.logit(0.625),
        jnp.log(10.0),
        np.pi/2,
        np.pi/2,
        10000.0
    ])
    
    # Perturb slightly
    init_params = init_params + jnp.array([0.1, -0.1, 0.1, 0.1, 0.1, 100.0])
    
    # -------------------------------------------------------------------------
    # Setup for Multiple Chains (Batching)
    # -------------------------------------------------------------------------
    # We want 4 chains. We can use the 'batch' dimension for this.
    n_chains = 4
    
    # Replicate Data: (1, 552) -> (4, 552)
    S_batch = jnp.tile(S_noisy, (n_chains, 1))
    
    # Replicate Params: (6,) -> (4, 6)
    # Add random perturbation per chain to ensure mixing check is valid
    k_init = jax.random.PRNGKey(999)
    perturbation = jax.random.normal(k_init, (n_chains, 6)) * 0.05
    # Scale perturbation for S0 differently
    perturbation = perturbation.at[:, 5].mul(1000.0) 
    
    init_params_batch = jnp.tile(init_params[None, :], (n_chains, 1)) + perturbation
    
    print(f"Starting MCMC with {n_chains} chains...")
    # Demo Mode: Faster sampling
    mcmc = MCMCInference(model_func, acq, sigma=sigma, n_samples=500, n_warmup=200)
    
    import time
    t0 = time.time()
    results = mcmc.fit(S_batch, init_params_batch, k_mcmc)
    t1 = time.time()
    
    # Results shape should be (n_chains, n_samples, n_params)
    samples = results['samples'] 
    print(f"Sample shape: {samples.shape}")
    
    # Flatten chains
    samples_flat = samples.reshape(-1, 6)
    
    # Calculate Speed
    total_samples = samples_flat.shape[0]
    duration = t1 - t0
    speed = total_samples / duration
    print(f"MCMC Finished in {duration:.2f}s. Speed: {speed:.1f} samples/sec")
    
    samples = samples_flat # Use flat samples for analysis
    
    # 5. Analysis
    # Transform samples back to physical
    f_iso_post = jax.nn.sigmoid(samples[:, 0])
    v_ic_post = jax.nn.sigmoid(samples[:, 1])
    kappa_post = jnp.exp(samples[:, 2])
    theta_post = samples[:, 3]
    phi_post = samples[:, 4]
    
    # Calculate global fractions for comparison
    w_ic_post = (1.0 - f_iso_post) * v_ic_post
    w_ec_post = (1.0 - f_iso_post) * (1.0 - v_ic_post)
    
    print("\n--- Results ---")
    print(f"Ground Truth: f_ic={w_ic:.3f}, f_iso={w_iso:.3f}, kappa={kappa_true:.1f}")
    
    print(f"Posterior Mean: f_ic={jnp.mean(w_ic_post):.3f}, f_iso={jnp.mean(f_iso_post):.3f}, kappa={jnp.mean(kappa_post):.1f}")
    
    ci_ic = jnp.percentile(w_ic_post, jnp.array([5, 95]))
    ci_iso = jnp.percentile(f_iso_post, jnp.array([5, 95]))
    ci_kappa = jnp.percentile(kappa_post, jnp.array([5, 95]))
    
    print(f"95% CI f_ic: [{ci_ic[0]:.3f}, {ci_ic[1]:.3f}]")
    print(f"95% CI f_iso: [{ci_iso[0]:.3f}, {ci_iso[1]:.3f}]")
    print(f"95% CI kappa: [{ci_kappa[0]:.1f}, {ci_kappa[1]:.1f}]")
    
    # 6. Plotting
    # Create a simple KDE plot for f_ic
    plt.figure(figsize=(10, 6))
    
    # f_ic distribution
    plt.hist(w_ic_post, bins=50, density=True, alpha=0.6, label='Posterior Hist')
    plt.axvline(w_ic, color='r', linestyle='--', linewidth=2, label='Ground Truth')
    plt.axvline(jnp.mean(w_ic_post), color='k', linestyle=':', linewidth=2, label='Rec. Mean')
    
    plt.title(f"Posterior Density for Intra-cellular Fraction (f_ic)\nSNR={SNR}, Speed={speed:.1f} samp/s")
    plt.xlabel("$f_{ic}$")
    plt.ylabel("Density")
    plt.legend()
    
    out_file = "mdt_reproduction_posterior.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    reproduction_workflow()
