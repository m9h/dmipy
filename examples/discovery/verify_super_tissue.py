import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.models.super_tissue_model import SuperTissueModel
from dmipy_jax.fitting.sparse import SparsityLMFitter
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

def verify_model_discovery():
    print("Verifying Automated Model Discovery...")
    
    # 1. Setup Acquisition
    # 2 shells + b0
    # Convert s/mm^2 to s/m^2 (SI)
    bvals = jnp.array([0.0] + [1000.0]*10 + [2000.0]*10) * 1e6
    # Random directions
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (21, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    # Needs timing for Cylinder
    acq = SimpleAcquisitionScheme(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=0.01, # 10 ms
        Delta=0.03  # 30 ms
    )
    
    # 2. Ground Truth: Pure Stick
    # Models: Stick, Ball, Zeppelin, Dot, Cylinder
    # Stick params: mu_stick, lambda_par
    
    true_stick_mu = jnp.array([1.57, 0.0]) # Theta, phi
    true_stick_lambda = 2.0e-9 
    
    # Instantiate Model for Generation
    stm = SuperTissueModel()
    
    # Parameter Construction
    # We need to construct the full parameter vector matching the order in STM.
    # We can inspect stm.parameter_names
    
    generated_params = []
    
    # We'll just define them manually based on known order:
    # 0: Stick (mu, lambda)
    # 1: Ball (iso)
    # 2: Zeppelin (mu, par, perp)
    # 3: Dot ()
    # 4: Cylinder (mu, par, diam)
    # Fractions: f0..f4
    
    # Values don't matter for 0-fraction comps, but let's be safe
    # Stick
    generated_params.extend([1.57, 0.0, 2.0e-9]) 
    # Ball
    generated_params.extend([3e-9])
    # Zeppelin
    generated_params.extend([0.0, 0.0, 1.0e-9, 0.5e-9])
    # Dot
    # No params
    # Cylinder
    generated_params.extend([0.0, 0.0, 1.5e-9, 5e-6])
    
    # Fractions (Pure Stick)
    generated_params.extend([1.0, 0.0, 0.0, 0.0, 0.0])
    
    gt_params = jnp.array(generated_params)
    
    # Generate Signal
    signal = stm(gt_params, acq)
    
    # Add Noise (SNR 50)
    sigma = 1.0 / 50.0
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, signal.shape) * sigma
    noisy_signal = jnp.sqrt((signal + noise)**2 + sigma**2) # Rician
    
    # 3. Fit
    # Fitter needs ranges
    param_ranges = []
    for name in stm.parameter_names:
        rn = stm.parameter_ranges[name]
        if isinstance(rn[0], list) or isinstance(rn[0], tuple): # spherical range for mu
            # Flatten spherical range? 
            # Fitters usually expect flat (min, max) list.
            # Dmipy ranges for mu are ([min_theta, min_phi], [max_theta, max_phi])??
            # Or ([theta_min, theta_max], [phi_min, phi_max])?
            # Let's check stm.parameter_ranges content.
            # Stick ranges: 'mu': ([0, pi], [-pi, pi])
            # We need to unpack.
            param_ranges.append((rn[0][0], rn[0][1])) # Theta
            param_ranges.append((rn[1][0], rn[1][1])) # Phi
        else:
            param_ranges.append(rn)
            
    # Careful: stm.parameter_names might list 'mu' once, but cardinality is 2.
    # So we need to iterate names and cardinality.
    
    flat_ranges = []
    for name in stm.parameter_names:
        card = stm.parameter_cardinality[name]
        rn = stm.parameter_ranges[name]
        
        if card > 1:
            # Special handling for spherical inputs which are tuple of tuples/lists
            if name.endswith('mu'):
                # ([0, pi], [-pi, pi])
                flat_ranges.append((rn[0][0], rn[0][1]))
                flat_ranges.append((rn[1][0], rn[1][1]))
            else:
                 # E.g. other array params? Assuming logic for mu covers it
                 pass
        else:
             flat_ranges.append(rn)
             
    # Initialize
    # Random init for all
    # Use multiplicative noise to preserve scale
    init_params = jnp.array(generated_params) * 0.8 
    # Add small positive epsilon to 0.0 values to avoid singularity if any
    init_params = jnp.where(init_params == 0.0, 1e-10, init_params)
    
    # Define Scales to Normalize Optimization Landscape
    # We want solver params to be ~O(1).
    # Diffusivities are ~1e-9 -> scale 1e-9
    # Diameter ~5e-6 -> scale 1e-6
    # Fractions/Angles ~1 -> scale 1.0
    
    scales = []
    for name in stm.parameter_names:
        # Heuristic scaling based on name
        if 'lambda' in name or 'diffusion' in name:
            scales.append(1e-9) # Diffusivity
        elif 'diameter' in name:
            scales.append(1e-6) # Diameter
        elif 'fraction' in name:
            scales.append(1.0) # Fraction
        else:
            scales.append(1.0) # Angles, etc.
            
        # Replicate for cardinality
        card = stm.parameter_cardinality[name]
        if card > 1:
            scales.extend([scales[-1]] * (card - 1))
            
    # Sparsity penalty tuning
    # If parameters are scaled to O(1), lambda can be O(0.01-0.1).
    # Previous lambda 0.02 is probably fine for O(1) fractions.
    
    # Fitter
    fitter = SparsityLMFitter(
        model_func=stm,
        parameter_ranges=flat_ranges,
        sparsity_lambda=0.05, # Slightly increased
        n_fractions=5,
        scales=scales
    )
    
    fitted_params, steps = fitter.fit(noisy_signal, acq, init_params)
    
    # 4. Analyze Results
    print(f"Optimization Steps: {steps}")
    
    # Debug: Check signal prediction
    fitted_prediction = stm(fitted_params, acq)
    mse = jnp.mean((fitted_prediction - noisy_signal)**2)
    print(f"Final MSE: {mse}")
    
    # Show High-B value predictions
    is_high_b = bvals > 1.9e9 # > 1900 s/mm^2
    print("\nSignal Comparison at High B-value (2000):")
    print(f"Index | Data | Prediction | Diff")
    for i in range(len(bvals)):
        if is_high_b[i]:
            print(f"{i:2d} | {noisy_signal[i]:.4f} | {fitted_prediction[i]:.4f} | {noisy_signal[i] - fitted_prediction[i]:.4f}")
            
    # Get Fractions (Last 5)
    fractions = fitted_params[-5:]
    print("Fitted Fractions (Stick, Ball, Zeppelin, Dot, Cylinder):")
    print(fractions)
    
    # Check Sparsity
    # Stick should be high, others zero.
    
    stick_frac = fractions[0]
    others = fractions[1:]
    
    assert stick_frac > 0.8, f"Stick fraction too low: {stick_frac}"
    assert jnp.all(others < 1e-4), f"Other fractions not zero enough: {others}"
    
    print("SUCCESS: Pure stick recovered, others driven to zero.")

if __name__ == "__main__":
    verify_model_discovery()
