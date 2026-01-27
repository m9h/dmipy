import jax
import jax.numpy as jnp
import ttax
import pytest
from dmipy_jax.tensor_train import tt_decompose_signal, global_local_bridge
# Attempt to mock compute_invariants_jax or ensure it runs.

def test_tt_decomposition():
    """
    Test basic TT decomposition wrapper.
    """
    # Create a small 5D volume
    # (B, 4, 4, 4, 15)
    # Using small size for speed
    shape = (1, 4, 4, 4, 15)
    data = jnp.ones(shape)
    
    tt = tt_decompose_signal(data, rank=2)
    
    # Check it is a TTTensor
    assert isinstance(tt, ttax.TT)
    
    # Reconstruct
    recon = ttax.full(tt)
    
    # Since rank=2 and data is rank 1 (ones), reconstruction should be perfect
    assert jnp.allclose(data, recon, atol=1e-4)

def test_global_local_bridge():
    """
    Test the bridge function.
    """
    # (B, X, Y, Z, Dir)
    # We need Dir=15 to match invariants logic L=4
    shape = (1, 2, 2, 2, 15)
    data = jax.random.normal(jax.random.PRNGKey(0), shape)
    
    tt = tt_decompose_signal(data, rank=2)
    
    # Bridge
    # Should output invariants (14 of them)
    res = global_local_bridge(tt, angular_dim_index=4)
    
    assert res.shape == (1, 2, 2, 2, 14) 
    
    # Check not NaN
    assert jnp.all(jnp.isfinite(res))
