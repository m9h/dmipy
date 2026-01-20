
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def demo_ball_and_stick():
    print("=== Dmipy-JAX: Ball & Stick Demo ===")
    
    # 1. Define Model
    # Ball (Isotropic) + Stick (Anisotropic)
    ball = Ball()
    stick = Stick()
    model = JaxMultiCompartmentModel([ball, stick])
    
    print("Model Parameters:", model.parameter_names)
    
    # 2. Synthetic Data
    # True Params
    f_stick_true = 0.6
    f_ball_true = 1 - f_stick_true
    mu_true = jnp.array([1.0, 0.0, 0.0])
    lambda_par_true = 1.7e-9
    lambda_iso_true = 3.0e-9
    
    params_true = {
        'partial_volume_0': jnp.array([f_ball_true]),
        'partial_volume_1': jnp.array([f_stick_true]),
        'lambda_iso': jnp.array([lambda_iso_true]),
        'mu': mu_true.reshape(1, 3), # (1, 3)
        'lambda_par': jnp.array([lambda_par_true])
    }
    
    # Acquisition
    bvals = jnp.tile(jnp.array([0, 1000, 2000, 3000]), 10) * 1e6
    bvecs = jax.random.normal(jax.random.PRNGKey(0), (40, 3))
    bvecs /= jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    acq = JaxAcquisition(bvals, bvecs)
    
    # Simulate
    # Note: model() usually handles batching if input is (N, ...)
    signal_clean = model(params_true, acq)
    
    # Add Noise
    key = jax.random.PRNGKey(1)
    noise = jax.random.normal(key, signal_clean.shape) * 0.02
    signal_noisy = jnp.abs(signal_clean + noise)
    
    print(f"Simulated Signal Shape: {signal_noisy.shape}")
    
    # 3. Fit
    print("Fitting...")
    # fit(acq, data)
    # data can be (N_vox, N_meas) or (N_meas,)
    
    fitted = model.fit(acq, signal_noisy)
    # Returns dictionary of fitted params
    
    print("Fitted Parameters:")
    for k, v in fitted.items():
        print(f"  {k}: {v[0]:.4e}")
        
    f_stick_fit = fitted['partial_volume_1'][0]
    print(f"\nGround Truth f_stick: {f_stick_true}")
    print(f"Recovered f_stick:    {f_stick_fit:.4f}")
    
    if abs(f_stick_fit - f_stick_true) < 0.1:
        print("SUCCESS: Recovery reasonable.")
    else:
        print("WARNING: Recovery check failed.")

if __name__ == "__main__":
    demo_ball_and_stick()
