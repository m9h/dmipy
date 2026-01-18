
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.biophysics.conductivity import (
    nernst_einstein_conductivity, 
    solve_voltage_field, 
    tdcs_objective_function, 
    _apply_variable_coefficient_laplacian,
    create_electrode_masks
)

def test_electrode_rasterization():
    positions = jnp.array([[2.0, 2.0, 2.0], [5.1, 5.9, 5.0]])
    shape = (10, 10, 10)
    
    # 1. Test without affine (voxels)
    masks = create_electrode_masks(positions, shape)
    assert masks.shape == (2, 10, 10, 10)
    assert masks[0, 2, 2, 2] == 1.0
    assert masks[1, 5, 6, 5] == 1.0 # Rounded 5.1->5, 5.9->6
    
    # 2. Test with affine (shift by 1 in x)
    # Physical = Voxel + [1, 0, 0]
    affine = jnp.eye(4)
    affine = affine.at[0, 3].set(1.0)
    
    # Position [3, 2, 2] physical -> [2, 2, 2] voxel
    pos_phys = jnp.array([[3.0, 2.0, 2.0]])
    masks_aff = create_electrode_masks(pos_phys, shape, affine=affine)
    assert masks_aff[0, 2, 2, 2] == 1.0

def test_nernst_einstein_units():
    # Test with standard values
    # D approx 3e-9 m^2/s (free water)
    # C approx 100 mol/m^3 (approx 100 mM)
    # T = 310.15
    # z = 1
    
    D_val = 3.0e-9
    C_val = 140.0 # Extracellular Na+, approx 140 mM = 140 mol/m^3
    
    D = jnp.eye(3) * D_val
    C = jnp.array(C_val)
    
    sigma = nernst_einstein_conductivity(D, C)
    
    # Check theoretical value
    # F = 96485
    # R = 8.314
    # T = 310.15
    # prefactor = F^2 / (RT) = (96485^2) / (8.314 * 310.15) approx 9310065225 / 2578 ~ 3.61e6
    # sigma = 3.61e6 * 140 * 3e-9 ~ 1.5 S/m
    
    scalar_sigma = sigma[0,0]
    assert scalar_sigma > 1.0 and scalar_sigma < 2.0
    
    # Check shape
    assert sigma.shape == (3, 3)

def test_poisson_solver_conservation():
    # Test on a small grid
    N = 10
    sigma = jnp.zeros((N, N, N, 3, 3))
    # Isotropic conductivity 1.0 everywhere
    for i in range(3):
        sigma = sigma.at[..., i, i].set(1.0)
        
    # Source: Source at (2,5,5), Sink at (7,5,5)
    source_map = jnp.zeros((N, N, N))
    source_map = source_map.at[2, 5, 5].set(100.0)
    source_map = source_map.at[7, 5, 5].set(-100.0)
    
    # Solve
    V = solve_voltage_field(sigma, source_map, voxel_size=0.001, maxiter=5000)
    
    # Check if potential is high at source and low at sink
    assert V[2, 5, 5] > V[7, 5, 5]
    
    # Verify discrete Laplacian matches source (approx)
    div_J = _apply_variable_coefficient_laplacian(V, sigma, voxel_size=0.001)
    # div(sigma grad V) = -source
    # so div_J should be approx -source
    
    # In interior points, error should be small
    diff = div_J - source_map
    
    # The solver uses the exact same operator, so residue should be close to 0 assuming CG converged
    # (Checking relative error)
    err = jnp.linalg.norm(diff) / jnp.linalg.norm(source_map)
    print(f"Poisson Solver Relative Residual: {err}")
    assert err < 1e-2

def test_tdcs_objective_gradient():
    # Smoke test for differentiability
    N = 6
    sigma = jnp.eye(3) * 1.0
    sigma_field = jnp.tile(sigma, (N, N, N, 1, 1))
    
    masks = jnp.zeros((2, N, N, N))
    masks = masks.at[0, 1, 3, 3].set(1.0) # Elec 1
    masks = masks.at[1, 4, 3, 3].set(1.0) # Elec 2
    
    roi = jnp.zeros((N, N, N))
    roi = roi.at[2:4, 3, 3].set(1.0)
    
    direction = jnp.array([1.0, 0.0, 0.0])
    
    curr = jnp.array([1.0, -1.0])
    
    # Func to grad
    def loss_fn(c):
        return tdcs_objective_function(c, masks, roi, direction, sigma_field)
    
    val, grad = jax.value_and_grad(loss_fn)(curr)
    
    assert not jnp.isnan(val)
    assert not jnp.any(jnp.isnan(grad))
    assert grad.shape == curr.shape

if __name__ == "__main__":
    test_nernst_einstein_units()
    test_poisson_solver_conservation()
    test_tdcs_objective_gradient()
    test_electrode_rasterization()
    print("All tests passed!")
