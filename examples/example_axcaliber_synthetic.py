
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time

from dmipy_jax.distributions.distributions import DD1Gamma
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.signal_models import cylinder_models
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def main():
    print("=== AxCaliber (Gamma-Distributed Cylinder) Synthetic Verification ===")
    
    # 1. Setup Acquisition Scheme
    # AxCaliber requires varying diffusion times (start varying big_delta) or high gradients.
    # Let's create a rich scheme.
    N_meas_per_shell = 30
    bvals_shell = 3000.0 * 1e6 # 3000 s/mm^2 in SI
    
    # Varying Delta (Diffusion Time)
    deltas = jnp.linspace(0.015, 0.050, 5) # 15ms to 50ms
    small_delta = 0.010 # 10ms
    
    bvals = []
    bvecs = []
    big_deltas = []
    small_deltas = []
    
    rng = np.random.default_rng(42)
    
    for D in deltas:
        # Generate N_meas directions
        vecs = rng.normal(size=(N_meas_per_shell, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        
        bvals.append(jnp.ones(N_meas_per_shell) * bvals_shell)
        bvecs.append(vecs)
        big_deltas.append(jnp.ones(N_meas_per_shell) * D)
        small_deltas.append(jnp.ones(N_meas_per_shell) * small_delta)
        
    bvals = jnp.concatenate(bvals)
    bvecs = jnp.concatenate(bvecs, axis=0)
    big_deltas = jnp.concatenate(big_deltas)
    small_deltas = jnp.concatenate(small_deltas)
    
    print(f"Acquisition: {len(bvals)} measurements.")
    print(f"  b-value: {bvals_shell/1e6} s/mm^2")
    print(f"  Deltas: {deltas} s")
    
    # 2. Define Ground Truth Model
    # Restricted Cylinder distributed over Diameter
    cyl = cylinder_models.RestrictedCylinder()
    gamma = DD1Gamma()
    
    # AxCaliber Model
    axcaliber = DistributedModel(cyl, gamma, target_parameter='diameter')
    
    # Parameters
    # Mean diameter ~ 5um. 
    # Gamma distribution: mu passed to model? No, Gamma takes alpha, beta.
    # Mean = alpha * beta. Variance = alpha * beta^2.
    # Let's target Mean=6um, Std=1um.
    # 6 = a*b
    # 1^2 = a*b^2 -> 1 = 6*b -> b = 1/6. a = 36.
    
    gt_params = {
        'lambda_par': 1.7e-9,  # 1.7 um^2/ms
        'mu': jnp.array([0.0, 0.0]), # Parallel to Z (theta=0)
        'alpha': 36.0,
        'beta': 1.0/6.0 * 1e-6, # Beta in meters? No, diameter is in meters. 
        # Wait, usually beta is scale. 
        # Let's set alpha=4, beta=1.5e-6 -> Mean = 6um.
        'big_delta': big_deltas,
        'small_delta': small_deltas
    }
    
    # Note on units: beta is scale parameter of diameter.
    # If diameter is ~1e-6 m. beta should be ~1e-6.
    gt_params['alpha'] = 10.0
    gt_params['beta'] = 0.5e-6 # Mean 5um
    
    # 3. Simulate Signal
    print("Simulating ground truth signal...")
    # Add batch dim for consistent handling if needed? Usually models handle (N,) acquisition.
    signal_noiseless = axcaliber(bvals, bvecs, **gt_params)
    
    # Add noise (Rician)
    SNR = 50.0
    noise_std = 1.0 / SNR
    noisy_real = signal_noiseless + rng.normal(scale=noise_std, size=signal_noiseless.shape)
    noisy_imag = rng.normal(scale=noise_std, size=signal_noiseless.shape)
    signal_noisy = jnp.sqrt(noisy_real**2 + noisy_imag**2)
    
    print(f"Simulated signal (SNR={SNR}).")

    # 4. Fit the Model
    print("Fitting model to synthetic data...")
    # Wrap in MultiCompartmentModel for convenient fitting API (uses OptimistixFitter if available)
    mcm = JaxMultiCompartmentModel([axcaliber])
    
    # We need an acquisition scheme object for the fit API generally, 
    # but the model.__call__ took raw arrays.
    # JaxMultiCompartmentModel.fit expects an AcquisitionScheme object which holds gradients etc.
    # BUT, AxCaliber requires varying deltas which are not always in standard .bval/.bvec/Scheme.
    # We need to construct a scheme that supports custom scan parameters?
    # Or just use the raw optimizer directly for this example to demonstrate correctness without API overhead.
    
    # Let's use simple least squares via JAX for clarity/transparency.
    
    import optax
    
    # Initial guess (raw unconstrained values will be optimized, so we init with inverse softplus of guess)
    import jax.nn as jnn
    
    def inv_softplus(x):
        return jnp.log(jnp.exp(x) - 1.0)

    # Initial physical values
    guesses = {
        'RestrictedCylinder_1_lambda_par': 1.5e-9,
        'RestrictedCylinder_1_mu_theta': 0.1, # treating mu components as scalars for simplicity
        'RestrictedCylinder_1_mu_phi': 0.1,
        'DD1Gamma_1_alpha': 5.0,
        'DD1Gamma_1_beta': 1.0e-6
    }
    
    # Init params for optimizer (unconstrained)
    # lambda, alpha, beta > 0. mu angles - unbounded (or periodic, but let's just optimize raw)
    params = {
        'lambda_raw': inv_softplus(guesses['RestrictedCylinder_1_lambda_par']),
        'mu_theta': guesses['RestrictedCylinder_1_mu_theta'],
        'mu_phi': guesses['RestrictedCylinder_1_mu_phi'],
        'alpha_raw': inv_softplus(guesses['DD1Gamma_1_alpha']),
        'beta_raw': inv_softplus(guesses['DD1Gamma_1_beta'])
    }
    
    print("\n=== Pre-Optimization Diagnostics ===")
    p_init = {
        'lambda_par': guesses['RestrictedCylinder_1_lambda_par'],
        'mu': jnp.array([params['mu_theta'], params['mu_phi']]),
        'alpha': guesses['DD1Gamma_1_alpha'],
        'beta': guesses['DD1Gamma_1_beta'],
        'big_delta': big_deltas,
        'small_delta': small_deltas
    }
    
    # 1. Check DD1Gamma Distribution
    print("Checking Distribution...")
    radii, prob = gamma(**p_init)
    print(f"Radii range: {jnp.min(radii)} to {jnp.max(radii)}")
    print(f"Prob range: {jnp.min(prob)} to {jnp.max(prob)}")
    print(f"Prob NaNs: {jnp.isnan(prob).any()}")
    print(f"Prob Area: {jnp.trapezoid(prob, x=radii)}")
    
    # 2. Check Base Model at Grid Points
    print("Checking Base Model evaluation...")
    test_d = radii[len(radii)//2]
    p_point = p_init.copy()
    p_point['diameter'] = test_d
    try:
        sig = cyl(bvals, bvecs, **p_point)
        print(f"Single point signal range: {jnp.min(sig)} to {jnp.max(sig)}")
        print(f"Signal NaNs: {jnp.isnan(sig).any()}")
    except Exception as e:
        print(f"Base model execution failed: {e}")
        
    # 3. Check Distributed Model
    print("Checking Distributed Model...")
    try:
        pred_init = axcaliber(bvals, bvecs, **p_init)
        print(f"Prediction range: {jnp.min(pred_init)} to {jnp.max(pred_init)}")
        print(f"Prediction NaNs: {jnp.isnan(pred_init).any()}")
    except Exception as e:
        print(f"Distributed model execution failed: {e}")
        
    print("=== End Diagnostics ===\n")
    
    # Optimization loop (simplified)
    # Lower learning rate for stability
    optimizer = optax.adam(learning_rate=0.05)
    
    def loss_fn(p):
        # Constrain (Softplus)
        current_p = {
            'lambda_par': jnn.softplus(p['lambda_raw']),
            'mu': jnp.array([p['mu_theta'], p['mu_phi']]),
            'alpha': jnn.softplus(p['alpha_raw']),
            'beta': jnn.softplus(p['beta_raw']),
            'big_delta': big_deltas,
            'small_delta': small_deltas
        }
        pred = axcaliber(bvals, bvecs, **current_p)
        return jnp.mean((pred - signal_noisy)**2)
    
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Running optimization (Adam) with constraints...")
    for i in range(1000):
        params, opt_state, loss_val = step(params, opt_state)
        if i % 100 == 0:
            a = jnn.softplus(params['alpha_raw'])
            b = jnn.softplus(params['beta_raw'])
            print(f"Iter {i}: Loss {loss_val:.2e}, Alpha {a:.2f}, Beta {b*1e6:.2f} um")
            
    print("\n=== Results ===")
    final_alpha = jnn.softplus(params['alpha_raw'])
    final_beta = jnn.softplus(params['beta_raw'])
    
    print("Parameter | Ground Truth | Recovered")
    print("-" * 40)
    print(f"Alpha     | {gt_params['alpha']:.4f}      | {final_alpha:.4f}")
    print(f"Beta (um) | {gt_params['beta']*1e6:.4f}      | {final_beta*1e6:.4f}")
    
    recovered_mean = final_alpha * final_beta
    gt_mean = gt_params['alpha'] * gt_params['beta']
    print(f"Mean Diam | {gt_mean*1e6:.4f} um    | {recovered_mean*1e6:.4f} um")
    
    if abs(recovered_mean - gt_mean)/gt_mean < 0.1:
        print("\nSUCCESS: Mean diameter recovered within 10%.")
    else:
        print("\nWARNING: Recovery deviation > 10%.")

if __name__ == "__main__":
    main()
