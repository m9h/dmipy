import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.composer import compose_models
from dmipy_jax.gaussian import G1Ball
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.fitting.optimization import VoxelFitter
import time

def test_fitting():
    print("Testing Voxel Fitting (Ball + Stick)...")
    
    # 1. Define Acquisition Scheme
    bvals = jnp.concatenate([jnp.zeros(1), jnp.ones(30)*1000, jnp.ones(30)*3000])
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(61, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    bvecs = jnp.array(vecs)
    
    acq = JaxAcquisition(
        bvalues=bvals, 
        gradient_directions=bvecs,
        delta=0.01,
        Delta=0.03
    )
    
    # 2. Instantiate Models
    ball = G1Ball()
    stick = C1Stick() 
    composite_model = compose_models([stick, ball])
    
    # 3. Ground Truth Parameters
    mu_gt = jnp.array([jnp.pi/2, 0.0])
    lambda_par_gt = 1.7e-3
    lambda_iso_gt = 3.0e-3
    frac_stick_gt = 0.6
    frac_ball_gt = 0.4
    
    params_gt = jnp.concatenate([
        mu_gt,
        jnp.array([lambda_par_gt]), 
        jnp.array([lambda_iso_gt]),
        jnp.array([frac_stick_gt, frac_ball_gt])
    ])
    
    print(f"Ground Truth: {params_gt}")
    signal_gt = composite_model(params_gt, acq)
    
    # 4. Define Fitter
    bounds = [
        (0.0, jnp.pi),         # theta
        (-jnp.pi, jnp.pi),     # phi
        (1e-5, 5e-3),          # lambda_par
        (1e-5, 5e-3),          # lambda_iso
        (0.0, 1.0),            # frac_stick
        (0.0, 1.0)             # frac_ball
    ]
    
    
    # 5. Define Fitter with Scaling
    # Scales: 
    # theta, phi -> 1.0
    # lambda_par, lambda_iso -> 1e-3 (so 1e-3 maps to 1.0)
    # fractions -> 1.0
    scales = [1.0, 1.0, 1e-3, 1e-3, 1.0, 1.0]
    
    fitter = VoxelFitter(
        composite_model, 
        bounds,
        solver_settings={'maxiter': 500, 'tol': 1e-8},
        scales=scales
    )
    
    # 5. Initial Guess
    init_params = jnp.array([
        jnp.pi/2 + 0.2, # theta approx pi/2
        0.2,            # phi approx 0
        1.5e-3,         # lambda_par approx 1.7
        2.5e-3,         # lambda_iso approx 3.0
        0.5,            # frac_stick approx 0.6
        0.5             # frac_ball approx 0.4
    ])
    
    print(f"Initial Guess: {init_params}")
    
    # 6. Run Fit
    start = time.time()
    fitted_params, state = fitter.fit(signal_gt, acq, init_params)
    fitted_params.block_until_ready()
    duration = time.time() - start
    
    print(f"Fit Complete in {duration*1000:.2f} ms")
    print("\nComparison:")
    print(f"{'Param':<15} {'GT':<15} {'Fitted':<15} {'Diff':<15}")
    names = ['theta', 'phi', 'lambda_par', 'lambda_iso', 'f_stick', 'f_ball']
    for n, gt, fit in zip(names, params_gt, fitted_params):
        print(f"{n:<15} {gt:<15.6f} {fit:<15.6f} {abs(gt-fit):<15.6f}")
        
    print(f"\nSolver Errors: {state.error}")
    print(f"Solver Iterations: {state.iter_num}")
    
    # Validation
    prediction = composite_model(fitted_params, acq)
    mse = jnp.mean((signal_gt - prediction)**2)
    print(f"Final MSE: {mse:.2e}")
    
    if mse < 1e-8:
        print("SUCCESS: Parameters recovered (MSE < 1e-8).")
    else:
        print("FAILURE: High MSE.")

if __name__ == "__main__":
    test_fitting()
