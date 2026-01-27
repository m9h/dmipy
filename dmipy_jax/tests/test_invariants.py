import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import pytest
from dmipy_jax.invariants import compute_invariants_jax

def test_rotational_invariance():
    """
    Test that the computed invariants are indeed invariant under rotation.
    """
    key = jax.random.PRNGKey(0)
    
    # Generate random SH coefficients (batch=1, dim=15 for L=0,2,4)
    # 0e (1) + 2e (5) + 4e (9) = 15
    coeffs = jax.random.normal(key, (2, 15))
    
    # Compute original invariants
    inv_orig = compute_invariants_jax(coeffs)
    
    # Apply a random rotation
    # We construct a rotation matrix/quaternion
    rot = e3nn.rand_matrix(jax.random.PRNGKey(1), ()) # Single random rotation
    # e3nn.IrrepsArray transform
    
    irreps = e3nn.Irreps("1x0e + 1x2e + 1x4e")
    arr = e3nn.IrrepsArray(irreps, coeffs)
    
    # Transform
    arr_rot = arr.transform_by_matrix(rot)
    coeffs_rot = arr_rot.array
    
    # Compute rotated invariants
    inv_rot = compute_invariants_jax(coeffs_rot)
    
    # Check close
    assert jnp.allclose(inv_orig, inv_rot, atol=1e-5), f"Max diff: {jnp.max(jnp.abs(inv_orig - inv_rot))}"

def test_differentiability():
    """
    Test that the invariant computation is differentiable.
    """
    coeffs = jnp.ones((1, 15))
    
    def loss(c):
        inv = compute_invariants_jax(c)
        return jnp.sum(inv**2)
    
    grad_fn = jax.grad(loss)
    grad = grad_fn(coeffs)
    
    assert jnp.all(jnp.isfinite(grad))
    assert grad.shape == coeffs.shape
