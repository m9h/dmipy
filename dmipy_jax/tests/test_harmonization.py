
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.harmonization.rish import RISHHarmonizer, fit_sh_coeffs, compute_rish_features
from dmipy_jax.utils.spherical_harmonics import cart2sphere, sh_basis_real

def test_fit_sh_coeffs_reconstruction():
    """Verify that SH fitting and reconstruction works for a band-limited signal."""
    lmax = 4
    n_dirs = 64
    
    # Generate random directions
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (n_dirs, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    # Generate a random SH signal (band-limited to lmax=4)
    # Number of even coeffs for lmax=4 is 1 (l=0) + 5 (l=2) + 9 (l=4) = 15
    n_coeffs = 15
    true_coeffs = jax.random.normal(key, (n_coeffs,))
    
    # Reconstruct signal manually
    x, y, z = bvecs.T
    r, theta, phi = cart2sphere(x, y, z)
    Y = sh_basis_real(theta, phi, lmax)
    signal = jnp.dot(Y, true_coeffs)
    
    # Fit
    fitted_coeffs = fit_sh_coeffs(signal, bvecs, lmax)
    
    # Check coefficient recovery
    assert jnp.allclose(fitted_coeffs, true_coeffs, atol=1e-5)
    
    # Check signal recovery
    reconstructed = jnp.dot(Y, fitted_coeffs)
    assert jnp.allclose(reconstructed, signal, atol=1e-5)

def test_compute_rish_features():
    """Verify RISH features are rotation invariant."""
    lmax = 4
    key = jax.random.PRNGKey(1)
    
    # Create coeffs
    n_coeffs = 15
    # l=0 (1), l=2 (5), l=4 (9)
    # Manually set coeffs so we know expected energy
    # l=0: c00=2.0 -> E0 = 2.0
    # l=2: c2m=[1,1,1,1,1] -> E2 = sqrt(5)
    # l=4: c4m=zeros -> E4 = 0
    
    coeffs = np.zeros(15)
    coeffs[0] = 2.0
    coeffs[1:6] = 1.0
    
    coeffs = jnp.array(coeffs)
    
    rish = compute_rish_features(coeffs, lmax)
    
    assert jnp.isclose(rish[0], 2.0)
    assert jnp.isclose(rish[1], jnp.sqrt(5.0))
    assert jnp.isclose(rish[2], 0.0)

def test_rish_harmonizer_learns_scale():
    """Verify Harmonizer learns the correct scale factor between two sites."""
    lmax = 4
    n_dirs = 128
    key = jax.random.PRNGKey(2)
    
    # Generate "Template" signal (Population average)
    bvecs = jax.random.normal(key, (n_dirs, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    n_coeffs = 15
    template_coeffs = jax.random.normal(key, (n_coeffs,))
    
    # Reconstruct signal
    x, y, z = bvecs.T
    r, theta, phi = cart2sphere(x, y, z)
    Y = sh_basis_real(theta, phi, lmax)
    template_signal = jnp.dot(Y, template_coeffs)
    
    # Site A (Reference) = Template
    ref_signal = template_signal
    ref_bvecs = bvecs
    
    # Site B (Target) = Scaled Template
    # Scale l=0 by 1.0, l=2 by 0.5, l=4 by 2.0
    scale_factors_true = jnp.array([1.0, 0.5, 2.0])
    
    # Apply inverse scaling to create target that needs harmonization
    # If Harmonizer learns Ref/Tgt, then Tgt = Ref / Scale
    # So if Scale=2, Tgt = Ref/2.
    # We want Harmonizer to recover [1.0, 0.5, 2.0]
    # So Tgt should be Ref / [1.0, 0.5, 2.0]
    
    # Expand scales
    expanded_scales = []
    for i, l in enumerate(range(0, lmax + 1, 2)):
        n_m = 2 * l + 1
        s_l = scale_factors_true[i]
        expanded_scales.append(jnp.full((n_m,), s_l))
    full_scales = jnp.concatenate(expanded_scales)
    
    tgt_coeffs = template_coeffs / full_scales
    tgt_signal = jnp.dot(Y, tgt_coeffs)
    tgt_bvecs = bvecs
    
    # Initialize Harmonizer
    harmonizer = RISHHarmonizer(lmax=lmax)
    
    # Fit (add batch dim)
    harmonizer = harmonizer.fit(
        ref_signal[None, ...], ref_bvecs, 
        tgt_signal[None, ...], tgt_bvecs
    )
    
    # Check learned scales
    # We used mean over 1 sample, so it should be exact
    print("True Scales:", scale_factors_true)
    print("Learned Scales:", harmonizer.scale_factors)
    
    assert jnp.allclose(harmonizer.scale_factors, scale_factors_true, rtol=1e-4)
    
    # Harmonize Target
    harmonized_signal = harmonizer(tgt_signal, tgt_bvecs)
    
    # Should match Reference
    assert jnp.allclose(harmonized_signal, ref_signal, atol=1e-4)
