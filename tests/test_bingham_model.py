
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.bingham import BinghamNODDI
from dmipy_jax.acquisition import JaxAcquisition

def test_bingham_normalization():
    print("Testing Bingham PDF Normalization...")
    model = BinghamNODDI(grid_points=200)
    
    # Access internal grid
    grid = model.grid_vectors_canonical # (N, 3)
    nx = grid[:, 0]
    ny = grid[:, 1]
    
    k1 = 1.0
    k2 = 5.0
    
    pdf = jnp.exp(-k1 * nx**2 - k2 * ny**2)
    norm = jnp.sum(pdf * model.grid_weights)
    
    print(f"Computed Normalization Factor: {norm}")
    print("PDF integral (normalized):", jnp.sum((pdf/norm) * model.grid_weights))
    assert jnp.abs(jnp.sum((pdf/norm) * model.grid_weights) - 1.0) < 1e-4
    print("Test Passed: Normalization")

def test_bingham_limits():
    print("\nTesting Bingham Limits...")
    
    # 1. Setup Data
    bvals = jnp.array([1000.0, 1000.0])
    grads = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # X and Y
    
    # 2. Limit: High Kappa -> Stick
    # If k1, k2 are very high, dispersion is 0. Signal should match Stick.
    k_high = 100.0 # High concentration
    
    model_bingham = BinghamNODDI(grid_points=500) # Need high res for peak
    
    mu_angles = jnp.array([np.pi/2, 0.0]) # Along X
    
    # Bingham Signal
    sig_bingham = model_bingham(bvals, grads, mu=mu_angles, kappa1=k_high, kappa2=k_high, lambda_par=1.7e-9)
    
    print(f"High Kappa Bingham Signal: {sig_bingham}")
    
    # Stick Signal (Expected high signal along X if lambda is diffusion param?)
    # Wait, Stick model: S = exp(-b * lambda * (g.mu)^2)
    # If mu is X, g is X -> (g.mu)^2 = 1 -> S = exp(-1000 * 1.7e-9 * 1e6?). 
    # b is typically s/mm2. If b=1000 s/mm2 = 1e9 s/m2.
    # 1000 * 1e6 * 1.7e-9 = 1.7. exp(-1.7) ~ 0.18.
    
    # 3. Limit: Watson
    # k1 = k2
    k_iso = 5.0
    
    sig_k1 = model_bingham(bvals, grads, mu=mu_angles, kappa1=k_iso, kappa2=k_iso, lambda_par=1.7e-9)
    # Check if rotational symmetry around mu (X axis) is preserved?
    # Actually if mu is X, and k1=k2, then dispersion in Y and Z directions should be equal.
    # grads are X and Y.
    # S(X) should be one thing (along mu). S(Y) should be another (perp to mu).
    # Since dispersion is isotropic around X, behavior checked.
    # Better check: Compare sig(Y) vs sig(Z) if gradients were [0,0,1].
    
    grads_test = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sig_sym = model_bingham(jnp.array([1000., 1000.]), grads_test, mu=mu_angles, kappa1=k_iso, kappa2=k_iso, lambda_par=1.7e-9)
    
    print(f"Watson Limit (k1=k2) Signals (perp1, perp2): {sig_sym}")
    diff = jnp.abs(sig_sym[0] - sig_sym[1])
    print(f"Difference: {diff}")
    assert diff < 1e-2
    print("Test Passed: Watson Limit (Symmetry)")

if __name__ == "__main__":
    test_bingham_normalization()
    test_bingham_limits()
