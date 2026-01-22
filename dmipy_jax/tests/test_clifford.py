import jax
import jax.numpy as jnp
from dmipy_jax.models.clifford import CliffordConv3d

def test_clifford_conv():
    key = jax.random.key(0)
    
    layer = CliffordConv3d(key)
    
    # Input: 8 spinor channels, 5x5x5 volume
    x = jax.random.normal(key, (8, 5, 5, 5))
    
    print("Running Clifford Convolution...")
    out = layer(x)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (8, 5, 5, 5)
    
    # Simple check for nan
    assert not jnp.isnan(out).any()
    
    print("CliffordConv3d test passed.")

if __name__ == "__main__":
    test_clifford_conv()
