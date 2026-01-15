
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

def cart2sphere(x, y, z):
    """
    Convert Cartesian coordinates to Spherical coordinates.
    Returns radius r, inclination theta [0, pi], azimuth phi [-pi, pi].
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Handle division by zero for origin
    r_safe = jnp.where(r == 0., 1., r)
    theta = jnp.arccos(jnp.clip(z / r_safe, -1.0, 1.0))
    phi = jnp.arctan2(y, x)
    return r, theta, phi

def factorial(n):
    return jnp.exp(gammaln(n + 1))

def associated_legendre(l, m, x):
    """
    Computes the Associated Legendre Polynomial P_l^m(x) using recurrence relations.
    This implementation is designed for small l and JIT compilation.
    Matches scipy.special.lpmv (with unnormalized definition).
    
    Args:
        l: degree (int)
        m: order (int, 0 <= m <= l)
        x: argument (float or array in [-1, 1])
    """
    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    # (2m-1)!! = factorial(2*m) / (2^m * factorial(m))
    
    # Precompute som constant factors could be optimized but JIT handles it.
    
    somx2 = jnp.sqrt(1 - x**2)
    
    # Compute P_m^m
    # double factorial (2m-1)!!
    # for m=0, fact=1. m=1, fact=1. m=2, fact=3. m=3, fact=15.
    
    # Using log gamma for stability/ease:
    # (2m-1)!! = Gamma(m + 1/2) * 2^m / sqrt(pi) -> actually simpler recurrence
    # P_0^0 = 1
    # P_1^1 = -sqrt(1-x^2)
    # P_2^2 = 3 * (1-x^2)
    # P_m^m = - (2m-1) * sqrt(1-x^2) * P_{m-1}^{m-1}
    
    pmm = jnp.ones_like(x)
    
    # We need a loop for P_m^m logic.
    # Since l and m are static integers, we can use python control flow for unrolling.
    
    # Calculate P_m^m
    if m > 0:
        val = 1.0
        # (2m-1)!! calculation: 1 * 3 * ... * (2m-1)
        for i in range(1, m + 1):
            val *= -(2 * i - 1)
        pmm = val * (somx2 ** m)

    if l == m:
        return pmm
    
    # Calculate P_{m+1}^m
    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    pmmp1 = x * (2 * m + 1) * pmm
    
    if l == m + 1:
        return pmmp1
    
    # Recurrence for remaining
    # (n - m) P_n^m = x(2n - 1) P_{n-1}^m - (n + m - 1) P_{n-2}^m
    # -> P_n^m = (x(2n - 1) P_{n-1}^m - (n + m - 1) P_{n-2}^m) / (n - m)
    
    p_prev2 = pmm
    p_prev1 = pmmp1
    p_curr = p_prev1 # dummy initialization
    
    for n in range(m + 2, l + 1):
        p_curr = (x * (2 * n - 1) * p_prev1 - (n + m - 1) * p_prev2) / (n - m)
        p_prev2 = p_prev1
        p_prev1 = p_curr
        
    return p_curr

def sph_harm_normalization(l, m):
    """
    Computes the normalization constant K_lm for Real Spherical Harmonics.
    Y_lm = K_lm * P_l^|m| * trig_term
    
    Orthonormalized such that integral over sphere is 1.
    """
    # K_lm = sqrt( (2l+1)/(4pi) * (l-m)! / (l+m)! )
    # Note: definition of P_l^m in scipy includes Condon-Shortley phase?
    # My associated_legendre implementation includes the (-1)^m phase for P_m^m.
    # This matches standard mathematical definition including Condon-Shortley.
    
    # Check if we need extra phase.
    # Usually Y_lm = sqrt(...) * P_l^m(cos theta) * exp(i m phi)
    
    # Using the standard physics convention (orthonormalized):
    # N = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    
    comb = (2 * l + 1) / (4 * jnp.pi) * jnp.exp(gammaln(l - m + 1) - gammaln(l + m + 1))
    return jnp.sqrt(comb)


def real_spherical_harmonic(l, m, theta, phi):
    """
    Computes Real Spherical Harmonic Y_lm(theta, phi).
    
    Args:
        l: Degree (int)
        m: Order (int) [-l, l]
        theta: Inclination [0, pi]
        phi: Azimuth [-pi, pi]
    """
    m_abs = abs(m)
    # Cosine of theta for Legendre
    x = jnp.cos(theta)
    
    plm = associated_legendre(l, m_abs, x)
    norm = sph_harm_normalization(l, m_abs)
    
    # Basis definition:
    # m > 0: sqrt(2) * N * P_l^m * cos(m phi)
    # m = 0: N * P_l^0
    # m < 0: sqrt(2) * N * P_l^|m| * sin(|m| phi)
    
    ys = norm * plm
    
    if m == 0:
        return ys
    elif m > 0:
        return jnp.sqrt(2) * ys * jnp.cos(m * phi)
    else:
        return jnp.sqrt(2) * ys * jnp.sin(m_abs * phi)


def sph_harm_basis(bvecs, max_order=6):
    """
    Constructs the Real Spherical Harmonic basis matrix for given directions.
    Only even orders l=0, 2, ..., max_order are used (symmetric diffusion).
    
    Args:
        bvecs: (N_dirs, 3) Gradient directions (cartesian).
        max_order: Maximum SH order (even int).
        
    Returns:
        B: (N_dirs, N_coeffs) Basis matrix.
    """
    _, theta, phi = cart2sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])
    
    basis_cols = []
    
    # Loop over even orders
    for l in range(0, max_order + 1, 2):
        for m in range(-l, l + 1):
            col = real_spherical_harmonic(l, m, theta, phi)
            basis_cols.append(col)
            
    return jnp.stack(basis_cols, axis=-1)


def fit_sh_coefficients(signal, basis_matrix):
    """
    Fits Spherical Harmonic coefficients using Least Squares.
    
    Args:
        signal: (..., N_dirs) DWI signal intensity. (Assumes last dim is directions)
        basis_matrix: (N_dirs, N_coeffs) SH basis matrix.
        
    Returns:
        coeffs: (..., N_coeffs) Estimated SH coefficients.
    """
    # Solve S = B * C  ->  C = pinv(B) * S
    # For broadcasting, we want C = S @ pinv(B).T or B_pinv @ S.T?
    
    # Precompute pinv (assumes unregularized least squares)
    # Binv = (B.T B)^-1 B.T
    # This is standard pseudo-inverse.
    
    # Using jnp.linalg.pinv is safe and handles rank deficiency if any.
    b_pinv = jnp.linalg.pinv(basis_matrix) # shape (N_coeffs, N_dirs)
    
    # Signal: (..., N_dirs). We want to dot with B_pinv.T (N_dirs, N_coeffs)
    # signal @ B_pinv.T -> (..., N_coeffs)
    return jnp.dot(signal, b_pinv.T)


def power_spectrum(coeffs, max_order=6):
    """
    Calculates the Rotational Invariant Power Spectrum (energy per band).
    P_l = sum_{m=-l}^l |c_lm|^2
    
    Args:
        coeffs: (..., N_coeffs) SH coefficients.
        max_order: Max order used in coefficients (must match coeffs size).
        
    Returns:
        invariants: (..., l_max/2 + 1) -> [P0, P2, P4, ...]
    """
    # Map coefficients to l, m
    # Coeffs are ordered: l=0(m=0), l=2(m=-2..2), l=4(m=-4..4), etc.
    
    invariant_list = []
    start_idx = 0
    
    for l in range(0, max_order + 1, 2):
        num_m = 2 * l + 1
        end_idx = start_idx + num_m
        
        # Extract c_lm for this l
        c_l = coeffs[..., start_idx:end_idx]
        
        # Compute power P_l
        # P_l = sum(c_lm^2) / (2l+1)? 
        # Wait, definition in prompt: P_l = sum |c_lm|^2.
        # Often normalized by 1/(2l+1) for "Rotationally Invariant Features" to be mean energy implies something else?
        # User defined: P_l = sum_{m=-l}^{l} |c_{lm}|^2.
        p_l = jnp.sum(c_l**2, axis=-1)
        
        invariant_list.append(p_l)
        start_idx = end_idx
        
    return jnp.stack(invariant_list, axis=-1)


def compute_invariants(signal, bvecs, max_order=6):
    """
    Main function to compute SH rotational invariants from DWI signal.
    
    Args:
        signal: (N_vox, N_dirs) Signal array.
        bvecs: (N_dirs, 3) B-vectors.
        max_order: Maximum SH order (default 6).
        
    Returns:
        invariants: (N_vox, N_orders) Power spectrum [P0, P2, P4, P6].
    """
    # 1. Construct Basis
    B = sph_harm_basis(bvecs, max_order=max_order)
    
    # 2. Fit Coefficients
    coeffs = fit_sh_coefficients(signal, B)
    
    # 3. Calculate Power Spectrum
    invariants = power_spectrum(coeffs, max_order=max_order)
    
    return invariants

# JIT compiled version for performance
compute_invariants_jit = jax.jit(compute_invariants, static_argnames=('max_order',))
