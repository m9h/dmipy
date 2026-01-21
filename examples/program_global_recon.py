
"""
Testing Program Part 2: Global Reconstruction (SCICO Showcase).

Demonstrates regularized Global AMICO reconstruction on "Synesthesia" data.
Goal: Show TV regularization cleaning up noise in the synthetic phantom.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# ensure benchmarks module is importable
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../benchmarks'))
from generate_synthetic_hcp import generate_synthetic_connectome

from dmipy_jax.inverse.global_amico import GlobalAMICOSolver
from dmipy_jax.signal_models import cylinder_models, gaussian_models
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def run_global_recon_program():
    print("=== Testing Program: Global AMICO Reconstruction ===")
    
    # 1. Generate Noisy Data (SNR=20 for visible noise)
    # 20x20x5 slice
    data, scheme, gt_params = generate_synthetic_connectome(shape=(20, 20, 5), snr=20)
    
    print("Data Shape:", data.shape)
    
    # 2. Define Dictionary (Models)
    # We use a simple 2-compartment model for AMICO: Stick + Ball
    # (Global AMICO usually works on linear dictionaries)
    # Stick with fixed diffusivity, Ball with fixed diffusivity.
    # Dictionary atoms: Sticks (varying orientation) + Ball.
    
    print("Setting up Dictionary...")
    stick = cylinder_models.C1Stick(lambda_par=1.7e-9)
    ball = gaussian_models.Ball(lambda_iso=3.0e-9)
    # For AMICO, we usually manual-construct the dictionary or use a helper.
    # `GlobalAMICOSolver` expects a `model` function that takes `dictionary_params`.
    
    # Let's use 32 directions for Sticks + 1 Ball.
    N_atoms = 33
    
    # Directions
    # Fibonacci sphere or similar
    t = np.linspace(0, np.pi, 32) # Dummy
    # Better: simple random for demo
    np.random.seed(99)
    dirs = np.random.randn(32, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Define model wrapper for solver
    # params: {'atom_idx': ...}
    
    class ModelWrapper:
        parameter_names = ('atom_idx',)
        
        def __init__(self, stick, ball, dirs):
            self.stick = stick
            self.ball = ball
            self.dirs_j = jnp.array(dirs)
            
        def __call__(self, bvals, gradient_directions, **kwargs):
            idx = kwargs['atom_idx']
            
            is_stick = idx < 32
            safe_idx = jnp.minimum(idx, 31)
            mu = self.dirs_j[safe_idx]
            
            s_stick = self.stick(bvals, gradient_directions, mu=mu)
            s_ball = self.ball(bvals, gradient_directions)
            
            return jnp.where(is_stick, s_stick, s_ball)

    wrapper_instance = ModelWrapper(stick, ball, dirs)

    # 3. Solver
    # 3. Solver
    print("Initializing GlobalAMICOSolver (SCICO)...")
    solver = GlobalAMICOSolver(
        model=wrapper_instance,
        acquisition=scheme,
        dictionary_params={'atom_idx': jnp.arange(N_atoms)}
    )
    
    # 4. Fit
    print("Running Global Fit (ADMM)...")
    data_crop = data[:, :, 2:3, :]
    
    coeffs = solver.fit_global(
        data_crop, 
        lambda_tv=0.1, 
        lambda_l1=0.0, 
        maxiter=50
    )
    
    print(f"Coeffs Shape: {coeffs.shape}")
    # (X, Y, Z, N_atoms)
    
    # 5. Visualize / Metrics
    # Compare smoothness of coeffs map vs independent fitting?
    # Calculate TV of result
    
    tv_val = jnp.sum(jnp.abs(jnp.diff(coeffs, axis=0))) + jnp.sum(jnp.abs(jnp.diff(coeffs, axis=1)))
    print(f"Result Total Variation: {tv_val:.4f}")
    
    # Compute Residual
    # Reconstruct signal: A * x
    # We can use solver.predict(coeffs) if implemented?
    # Or manual
    # Skip for now.
    
    print("[Test Complete: Global AMICO ran successfully]")

if __name__ == "__main__":
    run_global_recon_program()
