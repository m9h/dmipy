import jax
import jax.numpy as jnp
from jax_md import space
import numpy as np
import pytest
from dmipy_jax.core import particle_engine

def test_brownian_step_isotropic():
    """Verifies that isotropic diffusion matches MSD = 6*D*t in 3D."""
    key = jax.random.PRNGKey(0)
    N = 10000
    dim = 3
    positions = jnp.zeros((N, dim))
    displacement_fn, shift_fn = particle_engine.non_periodic_box()
    
    D = 2.0  # Diffusion coefficient
    dt = 0.01
    
    # Take one giant step or multiple small steps? 
    # Brownian motion is additive, so one step is sufficient to check distribution variance
    
    new_pos = particle_engine.brownian_step(key, positions, shift_fn, D, dt)
    
    displacements = new_pos - positions
    msd = jnp.mean(jnp.sum(displacements**2, axis=-1))
    
    expected_msd = 2 * dim * D * dt
    
    # Stochastic process, allow some margin
    # Standard error of mean of variance for normal dist is sigma^2 * sqrt(2/(N-1)) roughly?
    # Actually MSD = <r^2>. r_i ~ N(0, 2Ddt). r^2/ (2Ddt) ~ Chi-squared(dim)
    # Mean of Chi-sq(k) is k. Var is 2k.
    # So Var(r^2) = (2Ddt)^2 * 2*dim
    # SEM = sqrt(Var) / sqrt(N)
    # We check if we are within 5 sigma
    
    rtol = 0.1 # 10% error tolerance should be plenty for N=10000
    print(f"Isotropic: MSD={msd}, Expected={expected_msd}")
    assert jnp.isclose(msd, expected_msd, rtol=rtol)

def test_brownian_step_anisotropic_tensor():
    """Verifies that tensor diffusion produces correct covariance."""
    key = jax.random.PRNGKey(1)
    N = 20000
    dim = 3
    positions = jnp.zeros((N, dim))
    _, shift_fn = particle_engine.non_periodic_box()
    
    # Define a diagonal tensor first for easy checking
    # D_xx = 3.0, D_yy = 0.5, D_zz = 1.0
    D_diag = jnp.array([3.0, 0.5, 1.0])
    D_tensor = jnp.diag(D_diag)
    
    dt = 0.01
    
    new_pos = particle_engine.brownian_step(key, positions, shift_fn, D_tensor, dt)
    
    # Variance along each axis should be 2 * D_ii * dt
    var = jnp.var(new_pos, axis=0) # (3,)
    expected_var = 2 * D_diag * dt
    
    print(f"Anisotropic Diag Var: {var}, Expected: {expected_var}")
    assert jnp.allclose(var, expected_var, rtol=0.1)
    
    # Test off-diagonal correlation
    # Rotate the tensor
    theta = jnp.pi / 4
    c, s = jnp.cos(theta), jnp.sin(theta)
    # Rotation around Z
    R = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    D_rot = R @ D_tensor @ R.T
    
    new_pos_rot = particle_engine.brownian_step(key, positions, shift_fn, D_rot, dt)
    
    cov_matrix = jnp.cov(new_pos_rot, rowvar=False) # (3, 3)
    expected_cov = 2 * D_rot * dt
    
    print(f"Anisotropic Rotated Cov:\n{cov_matrix}\nExpected:\n{expected_cov}")
    assert jnp.allclose(cov_matrix, expected_cov, atol=1e-2, rtol=0.1)

def test_neighbor_list():
    """Verifies neighbor list construction handles periodic boundaries and cutoff."""
    box_size = 10.0
    displacement_fn, shift_fn = particle_engine.periodic_box(box_size)
    
    # Place two particles within cutoff, one outside
    # P1 at origin
    # P2 at (1, 0, 0) -> dist 1
    # P3 at (5, 0, 0) -> dist 5
    positions = jnp.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [5., 0., 0.]
    ])
    
    r_cutoff = 1.5
    
    nbr_fn = particle_engine.create_neighbor_list(displacement_fn, box_size, r_cutoff)
    
    nbr_list = nbr_fn.allocate(positions)
    
    # Neighbor list format is (max_neighbors, N) usually or dense padding
    # For Dense format in jax_md <= 0.2:
    #   idx.shape is (N, max_neighbors)
    #   We just check if particle 0 has particle 1 as neighbor and NOT particle 2
    
    neighbors_of_0 = nbr_list.idx[0]
    
    # Note: particle 0 is its own neighbor in some implementations or not, usually masked out or dist=0
    # jax-md usually does NOT include self if using standard neighbor lists, but let's check
    
    # We expect index 1 to be present. Index 2 should likely be N (masked) or not present.
    
    # Check distances
    # We might need to handle the fact that neighbor list size is dynamic/padding
    # Just checking boolean containment
    
    # Convert to numpy for easier `in` check if needed, but jax works
    print(f"Neighbors of 0: {neighbors_of_0}")
    
    # Depending on jax-md version, masked values are N or -1.
    # Assuming N=3, masked might be 3.
    
    assert 1 in neighbors_of_0
    assert 2 not in neighbors_of_0

if __name__ == "__main__":
    test_brownian_step_isotropic()
    test_brownian_step_anisotropic_tensor()
    test_neighbor_list()
    print("All tests passed!")
