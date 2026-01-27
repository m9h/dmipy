
import numpy as np
import jax
import jax.numpy as jnp
from dmipy_jax.bayesian.models import ball_and_sticks_ard
from dmipy_jax.bayesian.inference import fit_batch_vi
from dipy.core.geometry import sphere2cart

def test_recovery():
    print("Testing Bayesian ARD Recovery...")
    
    # 1. Generate Synthetic Data
    # 2 Crossing Fibers at 90 degrees
    # f1 = 0.3, f2 = 0.3, iso = 0.4
    bval_mag = 1000.0
    n_dir = 64
    theta = np.linspace(0, np.pi, n_dir)
    phi = np.linspace(0, 2*np.pi, n_dir)
    x, y, z = sphere2cart(1, theta, phi)
    bvecs = np.concatenate([np.zeros((1, 3)), np.stack([x, y, z], axis=1)], axis=0) # (N+1, 3)
    bvals = np.concatenate([np.zeros(1), np.ones(n_dir) * bval_mag]) # (N+1,)
    
    # Ground Truth
    S0_true = 1.0
    d_true = 0.002
    f1_true = 0.3
    f2_true = 0.3
    # v1 along x, v2 along y
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    
    # Generate signal manually
    # S = S0 * ( (1-f1-f2)*exp(-b*d) + f1*exp(-b*d*(g.v1)^2) + f2*exp(-b*d*(g.v2)^2) )
    E_ball = np.exp(-bvals * d_true)
    E_stick1 = np.exp(-bvals * d_true * (bvecs @ v1)**2)
    E_stick2 = np.exp(-bvals * d_true * (bvecs @ v2)**2)
    
    signal_clean = S0_true * ( (1 - f1_true - f2_true) * E_ball + f1_true * E_stick1 + f2_true * E_stick2 )
    
    # Add noise (Gaussian approximation)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, size=signal_clean.shape)
    data = signal_clean + noise
    
    # 2. Run Inference
    # Create batch of 1 for testing
    data_batch = jnp.array(data[None, :])
    bvals_j = jnp.array(bvals)
    bvecs_j = jnp.array(bvecs)
    
    key = jax.random.PRNGKey(0)
    
    # Fit with 3 fibers allowed
    res, losses = fit_batch_vi(
        key, 
        data_batch, 
        bvals_j, 
        bvecs_j, 
        n_fibers=3, 
        ard_weight=10.0, 
        num_steps=3000
    )
    
    # 3. Analyze Results
    print(f"S0 est: {res['S0'][0]}")
    print(f"d est: {res['d'][0]}")
    if 'sigma' in res:
        print(f"sigma est: {res['sigma'][0]}")
        
    f_est = res['f'][0]
    print(f"Estimated fractions: {f_est}")
    
    # Sort fractions
    f_sorted = np.sort(f_est)[::-1]
    print(f"Sorted fractions: {f_sorted}")
    
    # Checks
    # Top 1 should be around 0.3
    assert np.allclose(f_sorted[0], 0.3, atol=0.1), f"F1 mismatch: {f_sorted[0]}"
    
    # Sum of fractions should be ~ 0.6
    f_sum = np.sum(f_est)
    assert np.allclose(f_sum, 0.6, atol=0.1), f"Total fraction mismatch: {f_sum}"
    
    # Check S0 and d
    assert np.allclose(res['S0'][0], S0_true, atol=0.1), f"S0 mismatch: {res['S0'][0]}"
    assert np.allclose(res['d'][0], d_true, atol=0.0005), f"Diffusivity mismatch: {res['d'][0]}"
    
    print("Test PASSED: Recovered global parameters and fiber density.")

if __name__ == "__main__":
    test_recovery()
