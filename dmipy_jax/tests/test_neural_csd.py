import jax
import jax.numpy as jnp
import equinox as eqx
import healpy as hp
from dmipy_jax.models.neural_csd import NeuralCSD

def test_neural_csd():
    key = jax.random.key(0)
    nside = 4
    npix = hp.nside2npix(nside)
    l_max = 4
    n_sh = (l_max + 1) * (l_max + 2) // 2
    
    model = NeuralCSD(
        nside=nside,
        in_channels=1,
        hidden_channels=4,
        out_channels=4, # not used really, hidden flows to head
        l_max_out=l_max,
        num_layers=2,
        key=key
    )
    
    # Input: (C=1, V=npix, D=4, H=4, W=4)
    D, H, W = 4, 4, 4
    x = jax.random.normal(key, (1, npix, D, H, W))
    
    print("Running forward pass...")
    out = model(x)
    
    print(f"Output shape: {out.shape}")
    expected_shape = (D, H, W, n_sh)
    assert out.shape == expected_shape
    
    # Check for NaNs
    assert not jnp.isnan(out).any()
    
    print("NeuralCSD test passed.")

if __name__ == "__main__":
    test_neural_csd()
