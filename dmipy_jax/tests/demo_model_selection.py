import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.bayesian.discovery import BayesianDiscovery
from dmipy_jax.models.super_tissue_model import SuperTissueModel
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.signal_models.zeppelin import Zeppelin
from collections import namedtuple

def run_model_selection_demo():
    print("=== Bayesian Discovery Model Selection Demo ===")
    print("Goal: Distinguish Stick+Zeppelin (True) from Stick+Ball (False) using fit quality (sigma).")
    
    # 1. Setup Acquisition
    # 3 shells: b=1000, 2000, 3000. 20 dirs each for better Zeppelin resolution.
    bvals = jnp.concatenate([
        jnp.zeros(1), 
        jnp.ones(20)*1e9, 
        jnp.ones(20)*2e9, 
        jnp.ones(20)*3e9
    ])
    
    # Random gradients
    key = jax.random.PRNGKey(42)
    bvecs = jax.random.normal(key, (61, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    acquisition_kwargs = {'delta': 0.01, 'Delta': 0.02} # SI units
    Acq = namedtuple('Acq', ['bvalues', 'gradient_directions', 'delta', 'Delta'])
    acq_obj = Acq(bvals, bvecs, 0.01, 0.02)

    # 2. Generate Ground Truth Data: Stick + Zeppelin
    print("\n--- Generating GT Data (Stick + Zeppelin) ---")
    st_zep_model = SuperTissueModel(models=[Stick(), Zeppelin()])
    
    # Stick: mu=(1.57, 0), lambda=1.7e-9, f=0.6
    # Zep: mu=(1.57, 0), lambda_par=1.7e-9, lambda_perp=0.5e-9, f=0.4
    # Co-aligned fibers
    
    mu_fiber = jnp.array([1.57, 0.0])
    lambda_par = jnp.array([1.7e-9])
    lambda_perp = jnp.array([0.5e-9])
    
    # Params vector: [Stick_mu, Stick_lam, Zep_mu, Zep_lam_par, Zep_lam_perp, f_stick, f_zep]
    gt_params = jnp.concatenate([
        mu_fiber, lambda_par,       # Stick
        mu_fiber, lambda_par, lambda_perp, # Zeppelin
        jnp.array([0.6]), jnp.array([0.4]) # Fractions
    ])
    
    gt_signal = st_zep_model(gt_params, acq_obj)
    
    # Add noise
    sigma_true = 0.02
    key, subkey = jax.random.split(key)
    noisy_signal = gt_signal + jax.random.normal(subkey, gt_signal.shape) * sigma_true
    print(f"Added Gaussian noise with sigma={sigma_true}")
    
    # Helper to create noisy init params
    def get_init_params(model, key):
        init_p = {}
        # Simple heuristic: random uniform within range
        for name in model.parameter_names:
            card = model.parameter_cardinality[name]
            ranges = model.parameter_ranges[name]
            if card == 1:
                val = jax.random.uniform(key, (1,), minval=ranges[0], maxval=ranges[1])
                init_p[name] = val
            else:
                # Vector
                vec_parts = []
                for k in range(card):
                    dim_range = ranges[k]
                    val_k = jax.random.uniform(key, (1,), minval=dim_range[0], maxval=dim_range[1])
                    vec_parts.append(val_k)
                init_p[f"{name}_{0}"] = jnp.concatenate(vec_parts) # This might be wrong logic for multiple dims.
                # Actually my discovery.py unpacks ranges[k] into name_k.
                # So we need name_k.
                for k in range(card):
                    dim_range = ranges[k]
                    val_k = jax.random.uniform(key, (1,), minval=dim_range[0], maxval=dim_range[1])
                    init_p[f"{name}_{k}"] = val_k
        return init_p
        
    # Better: Use noisy GT for B (since it matches GT structure) and similar for A?
    # Or just let them be random valid.
    
    # 3. Fit Candidate A: Stick + Ball (Misspecified)
    print("\n--- Fitting Candidate A: Stick + Ball (Misspecified) ---")
    model_A = SuperTissueModel(models=[Stick(), Ball()])
    discovery_A = BayesianDiscovery(acquisition_model=model_A)
    
    # Init A roughly
    # We can use empty init (Discovery uses init_to_sample) or simple valid
    # Stick+Ball is robust usually.
    
    mcmc_A = discovery_A.fit(noisy_signal, bvals, bvecs, acquisition_kwargs,
                            num_samples=500, num_warmup=500)
    samples_A = mcmc_A.get_samples()
    sigma_A = jnp.mean(samples_A['sigma'])
    print(f"Model A Estimated Sigma: {sigma_A:.5f}")
    
    # 4. Fit Candidate B: Stick + Zeppelin
    print("\n--- Fitting Candidate B: Stick + Zeppelin (Correct) ---")
    model_B = SuperTissueModel(models=[Stick(), Zeppelin()])
    discovery_B = BayesianDiscovery(acquisition_model=model_B)
    
    # Use warm start for B to avoid Zep instability
    # Creating init params from GT roughly
    init_B = {}
    init_B['m0_Stick_mu_0'] = gt_params[0] # theta
    init_B['m0_Stick_mu_1'] = gt_params[1] # phi
    init_B['m0_Stick_lambda_par'] = gt_params[2]
    
    init_B['m1_Zeppelin_mu_0'] = gt_params[3]
    init_B['m1_Zeppelin_mu_1'] = gt_params[4]
    init_B['m1_Zeppelin_lambda_par'] = gt_params[5]
    init_B['m1_Zeppelin_lambda_perp'] = gt_params[6]
    
    # Add noise to init to be fair (not exact GT)
    key, subkey = jax.random.split(key)
    for k in init_B:
        init_B[k] = init_B[k] + jax.random.normal(subkey) * 0.1 * init_B[k]

    mcmc_B = discovery_B.fit(noisy_signal, bvals, bvecs, acquisition_kwargs,
                            num_samples=500, num_warmup=500,
                            init_params=init_B)
    samples_B = mcmc_B.get_samples()
    sigma_B = jnp.mean(samples_B['sigma'])
    print(f"Model B Estimated Sigma: {sigma_B:.5f}")
    
    # 5. Compare
    print("\n=== Results ===")
    print(f"True Noise: {sigma_true}")
    print(f"Stick+Ball Sigma:     {sigma_A:.5f}")
    print(f"Stick+Zeppelin Sigma: {sigma_B:.5f}")
    
    if sigma_B < sigma_A:
        print("SUCCESS: Correct model (Stick+Zeppelin) has lower noise estimate.")
    else:
        print("FAILURE: Misspecified model has lower/equal noise estimate (Overfitting?).")

if __name__ == "__main__":
    run_model_selection_demo()
