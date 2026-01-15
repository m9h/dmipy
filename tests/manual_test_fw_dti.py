
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.models.fw_dti import SpatiallyRegularizedFWDTI, fw_dti_predict

def test_fw_dti_run():
    # 1. Setup small grid
    shape = (4, 4, 4)
    n_voxels = np.prod(shape)
    
    # 2. Synthetic Parameters
    # S0 = 1000
    # f_iso = 0.3 (inside), 1.0 (water)
    # Tensor: pure stick x-direction -> D_xx large
    
    params_gt = jnp.zeros(shape + (8,))
    params_gt = params_gt.at[..., 0].set(1000.0) # S0
    params_gt = params_gt.at[..., 1].set(0.3) # f_iso
    
    # Tensor D ~ diag(2e-9, 0.2e-9, 0.2e-9)
    # L = diag(sqrt(2e-9), sqrt(0.2e-9), sqrt(0.2e-9))
    # c1, c3, c6
    c1 = jnp.sqrt(2.0e-9)
    c3 = jnp.sqrt(0.2e-9)
    c6 = jnp.sqrt(0.2e-9)
    
    params_gt = params_gt.at[..., 2].set(c1)
    params_gt = params_gt.at[..., 4].set(c3)
    params_gt = params_gt.at[..., 7].set(c6)
    
    # 3. Protocol
    # 3 shells: b=0, 1000, 2000 (s/mm^2) -> 0, 1e9, 2e9 s/m^2 approx
    # Actually b vals should be in SI. 1000 s/mm^2 = 1e6 * 1000 s/m^2 = 1e9.
    bvals = jnp.array([0., 1e9, 1e9, 1e9, 2e9, 2e9, 2e9])
    # Dirs
    sq2 = 1.0/jnp.sqrt(2)
    bvecs = jnp.array([
        [1., 0., 0.], # b0 - ignored usually or direction irrelevant
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [sq2, sq2, 0.],
        [sq2, 0., sq2],
        [0., sq2, sq2]
    ])
    
    # 4. Generate Signal
    data_synth = fw_dti_predict(params_gt, bvals, bvecs, d_iso=3.0e-9)
    
    # 5. Mask
    mask = jnp.ones(shape)
    
    # 6. Fit
    model = SpatiallyRegularizedFWDTI(data_synth, bvals, bvecs, mask)
    print("Starting optimization...")
    res = model.fit(lambda_reg=0.0) # No reg for perfect check
    
    print(f"Optimization Success: {res.state.success}")
    print(f"Final Loss: {res.state.fun_val}")
    
    # Check parameters
    params_est = res.params.reshape(shape + (8,))
    
    # Check f_iso
    f_iso_est = params_est[..., 1]
    print(f"Mean f_iso estimated: {jnp.mean(f_iso_est)}")
    print(f"Mean f_iso GT: 0.3")
    
    # Check difference
    mse_f = jnp.mean((f_iso_est - 0.3)**2)
    print(f"MSE f_iso: {mse_f}")
    
    assert mse_f < 0.05, "Approximation failed for f_iso"

if __name__ == "__main__":
    try:
        test_fw_dti_run()
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
