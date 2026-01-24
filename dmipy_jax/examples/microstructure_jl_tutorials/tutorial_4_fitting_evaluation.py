
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.fitting.optimization import OptimistixFitter, VoxelFitter

# We use the modeling framework to create a fit-able model
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def main():
    print("Tutorial 4: Fitting Evaluation")
    
    # 1. Setup Model (Stick + Ball)
    stick = C1Stick()
    ball = G1Ball()
    
    # Create MCM (handles parameters fitting interface)
    model = JaxMultiCompartmentModel([stick, ball])
    
    # Fix diffusivity to make problem easier (1D search over f?)
    # Or fit 3 parameters: theta, phi, f_stick (derived from f_stick/f_ball ratio)
    # The JaxMultiCompartmentModel fits ALL parameters of components + fractions.
    # Stick: mu (2), lambda_par (1)
    # Ball: lambda_iso (1)
    # Fractions: f_stick, f_ball
    # Total: 2+1 + 1 + 2 = 6 parameters.
    
    # We want to lock lambda_par and lambda_iso.
    # The current JaxMultiCompartmentModel ranges support allows bounds, but locking?
    # If range is [val, val], it's effectively locked?
    # Or we can subclass or use a custom function with `VoxelFitter`.
    
    # Let's use the object-oriented `JaxMultiCompartmentModel` and set tight bounds for fixed params.
    model.parameter_ranges['lambda_par_1'] = (1.7e-9, 1.7e-9) # Fixed
    model.parameter_ranges['lambda_iso_2'] = (3.0e-9, 3.0e-9) # Fixed
    # Note: names might have suffixes _1, _2
    # Let's check names
    print("Model Params:", model.parameter_names)
    
    # Adjust ranges
    # Stick is likely index 1 (or 0)
    # Ball is index 2 (or 1)
    
    # 2. Acquisition
    # Multi-shell
    bvals = jnp.kron(jnp.array([1000, 2000, 3000]) * 1e6, jnp.ones(10))
    key = jax.random.PRNGKey(42)
    bvecs = jax.random.normal(key, (30, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # 3. Simulate Ground Truth Data
    n_voxels = 100
    
    # GT parameters
    # theta: 0..pi/2
    gt_theta = jnp.linspace(0, jnp.pi/2, n_voxels)
    gt_phi = jnp.zeros(n_voxels)
    gt_lambda_par = jnp.full(n_voxels, 1.7e-9)
    gt_lambda_iso = jnp.full(n_voxels, 3.0e-9)
    gt_f_stick = jnp.full(n_voxels, 0.6)
    gt_f_ball = jnp.full(n_voxels, 0.4)
    
    # Construct parameter dictionary for simulation
    # Need to match model.parameter_names
    # Let's inspect names at runtime or assume standard
    # Check if 'mu' collisions -> mu_1?
    # Stick has 'mu', Ball has no 'mu'. So 'mu' might be unique if Ball doesn't have it.
    
    # For simulation, we'll manually call components first to be safe, 
    # then format for fitting.
    
    # Actually, let's just perform the fit on one simulated voxel repeatedly with noise
    # to evaluate BIAS and VARIANCE.
    
    n_trials = 200
    voxel_gt = {
        'mu': jnp.array([0.5, 1.0]), # theta, phi
        'lambda_par': 1.7e-9,
        'lambda_iso': 3.0e-9,
        'partial_volume_0': 0.6, # Stick
        'partial_volume_1': 0.4  # Ball
    }
    
    # Simulate Clean Signal
    clean_signal = model(voxel_gt, acq)
    
    # Add noise (Rician)
    sigma = 0.05
    key, subkey = jax.random.split(key)
    noise_r = jax.random.normal(subkey, (n_trials, len(bvals))) * sigma
    key, subkey = jax.random.split(key)
    noise_i = jax.random.normal(subkey, (n_trials, len(bvals))) * sigma
    
    noisy_data = jnp.sqrt((clean_signal + noise_r)**2 + noise_i**2)
    
    # 4. Fit
    print("Fitting 200 trials...")
    # fit() handles batching if data is (N_vox, N_meas)
    results = model.fit(acq, noisy_data, method="Levenberg-Marquardt", compute_uncertainty=False)
    
    # 5. Evaluate
    est_f_stick = results['partial_volume_0']
    
    bias = jnp.mean(est_f_stick) - voxel_gt['partial_volume_0']
    variance = jnp.var(est_f_stick)
    mse = jnp.mean((est_f_stick - voxel_gt['partial_volume_0'])**2)
    
    print(f"Ground Truth f_stick: {voxel_gt['partial_volume_0']}")
    print(f"Estimated Mean: {jnp.mean(est_f_stick):.4f}")
    print(f"Bias: {bias:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"MSE: {mse:.4f}")
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(est_f_stick, bins=20, alpha=0.7, label='Estimates')
    plt.axvline(voxel_gt['partial_volume_0'], color='r', linestyle='--', label='Ground Truth')
    plt.xlabel('Estimated f_stick')
    plt.ylabel('Count')
    plt.title(f'Tutorial 4: Fitting Evaluation (Bias={bias:.3f}, Var={variance:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('tutorial_4_output.png')
    print("Saved evaluation plot.")

if __name__ == "__main__":
    main()
