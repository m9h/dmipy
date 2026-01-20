
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.inverse.amico import AMICOSolver, calculate_mean_parameter_map
# from shapely.geometry import Point # Dummy import removed
# Using simple callable for model to avoid dependencies on specific signal models being perfect
# But let's try to mimic a Stick model structure

def dummy_stick(params, acquisition):
    # simple linear model: signal = diff * bval
    # params: {'diffusivity': float}
    bvals = acquisition['bvals']
    D = params['diffusivity']
    return jnp.exp(-bvals * D)

class TestAMICOSolver:
    def test_generating_kernels(self):
        # Setup acquisition
        acquisition = {'bvals': jnp.array([0., 1000., 2000., 3000.])}
        
        # Setup dictionary params
        # 3 atoms with Diffusivity = 1e-3, 2e-3, 3e-3
        dict_params = {
            'diffusivity': jnp.array([1e-3, 2e-3, 3e-3])
        }
        
        solver = AMICOSolver(dummy_stick, acquisition, dict_params)
        
        assert solver.dict_matrix.shape == (4, 3) # 4 measurements, 3 atoms
        
        # Check values manually
        # Atom 0 (D=1e-3): exp(-[0, 1, 2, 3])
        expected_col_0 = jnp.exp(-acquisition['bvals'] * 1e-3)
        assert jnp.allclose(solver.dict_matrix[:, 0], expected_col_0, atol=1e-5)

    def test_fit_sparse_recovery(self):
        # Setup acquisition
        bvals = jnp.linspace(0, 3000, 10)
        acquisition = {'bvals': bvals}
        
        # Dictionary: D from 0.5e-3 to 3.0e-3
        ds = jnp.linspace(0.5e-3, 3.0e-3, 6) # 6 atoms
        dict_params = {'diffusivity': ds}
        
        solver = AMICOSolver(dummy_stick, acquisition, dict_params)
        
        # Ground Truth: Only 2nd atom (index 1) has weight 1.0
        true_weights = jnp.zeros(6)
        true_weights = true_weights.at[1].set(1.0) # Atom corresponds to ds[1]
        
        # Generate data
        data = solver.dict_matrix @ true_weights
        
        # Fit
        # Use simple non-negative least squares logic (lambda=0 is fine if dictionary is well-conditioned enough)
        # But ADMM might need tuning. Let's try lambda=0 first (just non-negativity).
        weights_hat = solver.fit(data, lambda_reg=0.0, constrained=True)
        
        # Verify
        assert jnp.allclose(weights_hat, true_weights, atol=1e-2)
        assert jnp.all(weights_hat >= -1e-5) # Non-negativity

    def test_fit_mixture(self):
        bvals = jnp.linspace(0, 3000, 20)
        acquisition = {'bvals': bvals}
        ds = jnp.linspace(0.5e-3, 3.0e-3, 5) 
        dict_params = {'diffusivity': ds}
        solver = AMICOSolver(dummy_stick, acquisition, dict_params)
        
        # Mixture: 0.5 * Atom 0 + 0.5 * Atom 4
        true_weights = jnp.array([0.5, 0.0, 0.0, 0.0, 0.5])
        data = solver.dict_matrix @ true_weights
        
        weights_hat = solver.fit(data, lambda_reg=0.001, constrained=True)
        
        # With ADMM approximate solution, tolerance should be relaxed slightly
        assert jnp.allclose(weights_hat, true_weights, atol=0.05)
        
    def test_vmap_batch_fit(self):
        bvals = jnp.array([0., 1000., 2000.])
        acquisition = {'bvals': bvals}
        ds = jnp.array([1e-3, 2e-3])
        dict_params = {'diffusivity': ds}
        solver = AMICOSolver(dummy_stick, acquisition, dict_params)
        
        # Batch of 2 voxels
        # Vixel 0: Atom 0
        # Voxel 1: Atom 1
        w0 = jnp.array([1.0, 0.0])
        w1 = jnp.array([0.0, 1.0])
        
        d0 = solver.dict_matrix @ w0
        d1 = solver.dict_matrix @ w1
        
        data_batch = jnp.stack([d0, d1]) # [2, 3]
        
        weights_batch = solver.fit(data_batch, lambda_reg=0.0, constrained=True)
        
        assert weights_batch.shape == (2, 2)
        assert jnp.allclose(weights_batch[0], w0, atol=1e-2)
        assert jnp.allclose(weights_batch[1], w1, atol=1e-2)

    def test_calculate_mean_parameter(self):
        # Setup: 2 atoms. Atom 0: d=1. Atom 1: d=5.
        dict_params = {'diffusivity': jnp.array([1.0, 5.0])}
        
        # Case 1: Pure Atom 0 -> Mean=1.0
        w1 = jnp.array([1.0, 0.0])
        m1 = calculate_mean_parameter_map(w1, dict_params, 'diffusivity')
        assert jnp.allclose(m1, 1.0)
        
        # Case 2: Equal mix -> Mean=3.0
        w2 = jnp.array([0.5, 0.5])
        m2 = calculate_mean_parameter_map(w2, dict_params, 'diffusivity')
        assert jnp.allclose(m2, 3.0)
        
        # Case 3: Batch
        w_batch = jnp.stack([w1, w2])
        m_batch = calculate_mean_parameter_map(w_batch, dict_params, 'diffusivity')
        assert jnp.allclose(m_batch, jnp.array([1.0, 3.0]))

if __name__ == "__main__":
    # If running directly
    t = TestAMICOSolver()
    t.test_generating_kernels()
    t.test_fit_sparse_recovery()
    t.test_fit_mixture()
    t.test_vmap_batch_fit()
    t.test_calculate_mean_parameter()
    print("All tests passed manually.")
