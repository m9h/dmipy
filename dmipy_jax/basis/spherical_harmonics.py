import jax.numpy as jnp
from jax.scipy.special import sph_harm_y

def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates to Spherical coordinates.
    Returns:
        r, theta, phi
        theta: Polar angle [0, pi] (from z-axis)
        phi: Azimuthal angle [0, 2pi] (from x-axis)
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Clip r to avoid division by zero gradients at origin if any
    r = jnp.where(r == 0, 1e-12, r)
    
    # theta = arccos(z/r)
    theta = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))
    
    # phi = arctan2(y, x)
    phi = jnp.arctan2(y, x)
    
    # Map phi to [0, 2pi] for consistency if needed, but sph_harm handles [-pi, pi]
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)
    
    return r, theta, phi

def real_sph_harm(l, m, theta, phi):
    """
    Compute Real Spherical Harmonic Y_{lm}(theta, phi).
    
    Physics convention:
    theta: Polar angle [0, pi]
    phi: Azimuthal angle [0, 2pi]
    
    m > 0: sqrt(2) * Re(Y_l^m)
    m < 0: sqrt(2) * Im(Y_l^{|m|})
    m = 0: Y_l^0
    """
    # Use sph_harm_y (JAX > 0.4.1)
    # JAX sph_harm_y signature: (n, m, theta, phi, n_max)
    # n: degree (l)
    # m: order
    # theta: Polar Angle [0, pi]
    # phi: Azimuthal Angle [0, 2pi]
    
    # We must treat l and m as arrays because JAX sph_harm_y fails on scalars with len()
    l_arr = jnp.array([l])
    m_abs_arr = jnp.array([jnp.abs(m)])
    
    # Pass n_max=l to be safe and efficient
    # Returns broadcasted shape (N,)
    # Note: jax.scipy.special.sph_harm_y / sph_harm follows scipy convention: (m, n, theta, phi)
    # where m is order, n is degree (l).
    # We must pass (m_abs_arr, l_arr).
    Y = sph_harm_y(l_arr, m_abs_arr, theta, phi, n_max=l)
    
    # Check m
    c = jnp.sqrt(2)
    
    y_real = jnp.where(
        m == 0,
        jnp.real(Y),
        jnp.where(
            m > 0,
            c * jnp.real(Y),
            c * jnp.imag(Y) # Y is computed with abs(m)
        )
    )
    return y_real

def sh_basis(gradients, max_order):
    """
    Generate Real Spherical Harmonics Basis Matrix.
    
    Args:
        gradients: (N, 3) array of gradient directions (x, y, z).
        max_order: Maximum SH order (must be even for symmetric dMRI).
        
    Returns:
        (N, N_coeffs) basis matrix. 
        N_coeffs = 0.5 * (max_order + 1) * (max_order + 2)
    """
    x, y, z = gradients[:, 0], gradients[:, 1], gradients[:, 2]
    r, theta, phi = cart2sph(x, y, z)
    
    basis = []
    
    # Loop over even orders
    for l in range(0, max_order + 1, 2):
        # Loop over m from -l to l
        for m in range(-l, l + 1):
            # Compute Y_lm
            # Note: We need correct m handling. 
            # real_sph_harm handles m logic internally using Y_l^|m|
            # But we need to pass |m| to sph_harm if we do it inside.
            # My real_sph_harm function calls sph_harm(abs(m), l, ...) so it is fine.
            
            basis_lm = real_sph_harm(l, m, theta, phi)
            basis.append(basis_lm)
            
    # Stack: (N_coeffs, N_samples) -> Transpose to (N_samples, N_coeffs)
    basis_matrix = jnp.stack(basis, axis=-1)
    
    return basis_matrix
