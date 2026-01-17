
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.composer import compose_models
from dmipy_jax.fitting.optimization import OptimistixFitter

def verify_optimistix_fitter():
    print("Verifying OptimistixFitter...")
    
    # 1. Setup Acquisition
    # Create a simple acquisition with 2 shells + b0
    bvalues = jnp.array([0.0] + [1000.0] * 6 + [2000.0] * 6)
    # Simple gradient directions (random-ish)
    vectors = np.random.randn(13, 3)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gradient_directions = jnp.array(vectors)
    
    acquisition = JaxAcquisition(
        bvalues=bvalues,
        gradient_directions=gradient_directions
    )
    
    # 2. Setup Model
    stick = Stick()
    # Compose models to get the fit-ready function signature: func(params, acquisition)
    model_func = compose_models([stick])
    
    # Stick params: mu (theta, phi), lambda_par
    # Composite params: [mu_theta, mu_phi, lambda_par, fraction]
    # Let's define Ground Truth
    gt_theta = 1.0 # rad
    gt_phi = 0.5   # rad
    # Diffusivity in mm^2/s (approx 1.7e-3 mm^2/s = 1.7 um^2/ms)
    gt_lambda = 1.7e-3 
    gt_frac = 1.0 # volume fraction
    
    gt_params = jnp.array([gt_theta, gt_phi, gt_lambda, gt_frac])
    
    # 3. Generate Synthetic Data
    data = model_func(gt_params, acquisition)
    
    # 4. Setup Fitter
    # Parameter ranges (min, max) for scaling
    # theta: [0, pi], phi: [-pi, pi], lambda: [0.1e-3, 3e-3], frac: [0, 1]
    param_ranges = [
        (0.0, jnp.pi),
        (-jnp.pi, jnp.pi),
        (0.1e-3, 3e-3),
        (0.0, 1.0)
    ]
    
    # Scales for better convergence (Optimistix works best with O(1) params)
    scales = [1.0, 1.0, 1e-3, 1.0]
    
    fitter = OptimistixFitter(
        model_func=model_func,
        parameter_ranges=param_ranges,
        scales=scales
    )
    
    # 5. Fit
    # Perturb GT for initial guess
    init_params = jnp.array([gt_theta + 0.1, gt_phi + 0.1, 1.5e-3, 0.9])
    
    fitted_params, result = fitter.fit(data, acquisition, init_params)
    
    print(f"Ground Truth: {gt_params}")
    print(f"Fitted Params: {fitted_params}")
    
    # 6. Verify
    # Check if fitted params are close to GT
    # Note: Orientation (mu) might have antipodal symmetry issues, but for small perturbation it should be close.
    # Lambda and fraction should be very close.
    
    error = jnp.linalg.norm(fitted_params - gt_params)
    print(f"L2 Error: {error}")
    
    if error < 1e-4: # Stricter than typical for 32-bit float but reasonable for clean synthetic data
        print("SUCCESS: OptimistixFitter converged to ground truth.")
    else:
        print("FAILURE: OptimistixFitter failed to converge.")
        exit(1)

if __name__ == "__main__":
    verify_optimistix_fitter()
