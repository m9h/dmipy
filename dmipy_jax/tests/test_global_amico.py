
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.sphere_models import SphereGPD
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme as JaxAcquisition
from dmipy_jax.inverse.global_amico import GlobalAMICOSolver

def test_global_amico_solver_smoothness():
    """
    Test that Global AMICO Solver runs and TV regularization induces smoothness.
    """
    # 1. Setup Model and Acquisition
    # Stick with fixed diffusivity
    stick = Stick(lambda_par=1.7e-9)
    # Sphere with fixed diffusivity and diameter
    sphere = SphereGPD(diffusion_constant=3e-9, diameter=10e-6)
    model = JaxMultiCompartmentModel([stick, sphere])
    
    # Simple Acquisition
    bvals = jnp.array([0.0] + [1000.0] * 6)
    bvecs = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0]
    ])
    acq = JaxAcquisition(bvals, bvecs, delta=0.01, Delta=0.02)
    
    # 2. Setup Dictionary
    # Grid for Stick mu: 6 directions (axes)
    # Grid for Sphere: fixed parameters in init usually, but let's assume default
    # Note: Stick needs orientation 'mu'.
    mu_grid = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    dictionary_params = {
        'mu': mu_grid # Stick orientation
    }
    
    # Initialize Solver
    solver = GlobalAMICOSolver(model, acq, dictionary_params)
    
    # 3. Create Synthetic Data (10x10x10)
    shape = (10, 10, 10)
    n_meas = len(bvals)
    
    # Create a "block" of signal in the center
    # Background: Signal A (e.g. Sphere dominated)
    # Block: Signal B (e.g. Stick dominated)
    
    # Get atoms to construct signal
    # Solver dictionary shape: (N_meas, N_atoms)
    # Atoms: [Stick_x, Stick_y, Stick_z, Sphere] (likely order)
    # Let's verify atom count
    n_atoms = solver.dict_matrix.shape[1]
    # Stick has 3 atoms (from grid). Sphere has 1 atom (default parameters).
    # Total 4.
    
    # True coefficients
    x_true = jnp.zeros(shape + (n_atoms,))
    
    # Background (everywhere): Sphere = 1.0
    x_true = x_true.at[..., 3].set(1.0)
    
    # Block (center): Stick_x = 1.0
    x_true = x_true.at[3:7, 3:7, 3:7, 0].set(1.0)
    
    # Generate data Y = X @ Phi^T
    Y_clean = jnp.dot(x_true, solver.dict_matrix.T)
    
    # Add Noise
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, Y_clean.shape) * 0.1
    Y_noisy = Y_clean + noise
    
    # 4. Fit with Lambda TV = 0
    print("Fitting with TV=0...")
    x_no_tv = solver.fit_global(
        Y_noisy, 
        lambda_tv=0.0, 
        lambda_l1=0.001, 
        maxiter=30, # Low iter for test speed
        display=False
    )
    
    # 5. Fit with Lambda TV = 0.5
    print("Fitting with TV=0.5...")
    x_tv = solver.fit_global(
        Y_noisy, 
        lambda_tv=0.5, 
        lambda_l1=0.001, 
        maxiter=30,
        display=False
    )
    
    # 6. Verify Smoothness
    # Calculate Gradient Norm for the Stick_x coefficient (index 0)
    def tv_norm(x_vol):
        grads = jnp.gradient(x_vol)
        return sum([jnp.sum(jnp.abs(g)) for g in grads])
        
    tv_score_no = tv_norm(x_no_tv[..., 0])
    tv_score_yes = tv_norm(x_tv[..., 0])
    
    print(f"TV Score (No Reg): {tv_score_no}")
    print(f"TV Score (With Reg): {tv_score_yes}")
    
    # Assert
    assert tv_score_yes < tv_score_no, "TV Regularization should reduce Total Variation score"
    
    # Verify non-negativity
    assert jnp.all(x_tv >= -1e-5), "Coefficients should be non-negative (roughly)"

