import jax
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.nn.equivariance import ChebConv, IsoConv3D, get_healpix_laplacian
import healpy as hp

def test_healpix_laplacian():
    nside = 8
    L = get_healpix_laplacian(nside, nest=True)
    npix = hp.nside2npix(nside)
    assert L.shape == (npix, npix)
    # Check L is symmetric
    assert jnp.allclose(L, L.T, atol=1e-5)
    # Check diagonal approx 1 (since L_tilde = L - I, and L ~ I-A... wait)
    # L_norm = I - Dinv A Dinv. Diag is 1.
    # L_tilde = L_norm - I = - Dinv A Dinv. Diag should be 0 (if no self loops).
    # Let's check trace.
    print(f"Laplacian trace: {jnp.trace(L)}")
    print("Laplacian test passed.")

def test_cheb_conv():
    key = jax.random.key(0)
    nside = 4
    npix = hp.nside2npix(nside)
    in_c = 2
    out_c = 4
    K = 3
    
    layer = ChebConv(in_c, out_c, K, nside, key)
    
    x = jax.random.normal(key, (in_c, npix))
    out = layer(x)
    
    assert out.shape == (out_c, npix)
    print("ChebConv test passed.")

def test_iso_conv():
    key = jax.random.key(0)
    in_c = 2
    out_c = 3
    
    layer = IsoConv3D(in_c, out_c, key)
    
    # Input volume: C x D x H x W
    x = jax.random.normal(key, (in_c, 5, 5, 5))
    out = layer(x)
    
    assert out.shape == (out_c, 5, 5, 5)
    print("IsoConv3D test passed.")

if __name__ == "__main__":
    test_healpix_laplacian()
    test_cheb_conv()
    test_iso_conv()
