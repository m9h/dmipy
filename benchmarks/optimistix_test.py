import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.fitting.optimization import OptimistixFitter

# 1. Setup Acquisition (SI Units: s/m^2)
# 1000 s/mm^2 = 1e9 s/m^2
bvalues = jnp.array([0.0, 1e9, 1e9, 1e9, 2e9, 2e9, 2e9])
# 1 b0, 3 b1000 (x,y,z), 3 b2000 (x,y,z)
bvecs = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])
acq = JaxAcquisition(bvalues=bvalues, gradient_directions=bvecs)

# 2. Setup Model (Stick)
stick = C1Stick()
# Parameters: [mu_theta, mu_phi, lambda_par] (Cardinality: mu=2, lambda_par=1)

print("C1Stick Params:", stick.parameter_names)
print("C1Stick Cardinality:", stick.parameter_cardinality)

# True parameters
# Orientation along X-axis: theta=pi/2, phi=0
# Diffusivity = 1.7e-9 m^2/s
true_params_flat = jnp.array([jnp.pi/2, 0.0, 1.7e-9])

def model_wrapper(params, acq):
    # Wrapper to unpack params into kwargs for the model
    # params = [theta, phi, lambda_par]
    mu = params[:2]
    lambda_par = params[2]
    return stick(bvals=acq.bvalues, gradient_directions=acq.gradient_directions, mu=mu, lambda_par=lambda_par)

# Generate Data
data = model_wrapper(true_params_flat, acq)
print("Synthetic Data:", data)

# 3. Fit with Optimistix
# Initial guess: Offset from X-axis to avoid orthogonal saddle point
# True phi=0.0. Init phi=0.4 (approx 23 degrees)
init_params = jnp.array([jnp.pi/2, 0.4, 1.0e-9])
scales = jnp.array([1.0, 1.0, 1e-9]) # Scaling angles by 1 is fine, diffusivity needs scaling

# Ranges (Ignored by current OptimistixFitter but passed for compat)
ranges = [(0, jnp.pi), (-jnp.pi, jnp.pi), (0.1e-9, 3e-9)]

fitter = OptimistixFitter(model_wrapper, ranges, scales=scales)
fitted_params, result = fitter.fit(data, acq, init_params)

print("\n--- Optimistix Results ---")
print("Init Params:", init_params)
print("Fitted Params:", fitted_params)
print("True Params:", true_params_flat)
print("Result Code:", result)

# Verify Signal Reconstruction
reconstructed_signal = model_wrapper(fitted_params, acq)
mse = jnp.mean((data - reconstructed_signal)**2)
print("MSE:", mse)

# Check parameters (account for periodicity/ambiguity if needed)
# mu vs -mu shouldn't happen with stick as it's antipodally symmetric in signal but parameters might drift?
# Actually C1Stick implementation converts to cartesian, so signal is symmetric.
# But parameters (theta, phi) might land on equivalent representation.
# Signal MSE is the robust check.

assert mse < 1e-10, f"MSE too high: {mse}"
print("VERIFICATION PASSED")
