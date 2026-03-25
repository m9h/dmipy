
import jax
import jax.numpy as jnp
import sympy
from sympy.abc import theta, phi
from sympy import Ynm, integrate, sin, conjugate, N, pi
import numpy as np
from functools import lru_cache

# We use SymPy to compute the Gaunt coefficients analytically.
# G_{l1,m1, l2,m2, L,M} = Integral( Y_{l1,m1} * Y_{l2,m2} * conj(Y_{L,M}) )
# But since we work with REAL SH in CSD, we need the Real Gaunt Tensor.
# For simplicity in this Agent 1 implementation, we will perform the mapping:
# FOD = P^2 where P is a Real SH polynomial.
# This ensures non-negativity.

def get_real_sh_basis_sympy(lmax):
    """
    Returns a list of SymPy expressions for Real Spherical Harmonics up to lmax.
    Order: l=0, l=2... (Symmetric only for dMRI CSD).
    """
    # Standard CSD assumes antipodal symmetry -> Even orders only.
    basis = []
    indices = []
    
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            # SymPy Ynm is Complex.
            # Real basis definition:
            # m > 0: 1/sqrt(2) (Y_l^{-m} + (-1)^m Y_l^m) ? No, standard definition:
            # P is Legendre.
            # Real SH:
            # m = 0: Y_l^0
            # m > 0: sqrt(2) * (-1)^m * Re(Y_l^m)
            # m < 0: sqrt(2) * (-1)^m * Im(Y_l^m) (where m is positive here? No, let's use standard)
            
            # Let's rely on sympy.functions.special.spherical_harmonics.Znm (Real SH)?
            # SymPy Znm exists! Znm(n, m, theta, phi)
            
            ylm = sympy.Znm(l, m, theta, phi)
            basis.append(ylm)
            indices.append((l, m))
            
    return basis, indices

def compute_sos_tensor(lmax_root, lmax_fod):
    """
    Computes the tensor T such that:
    c_{fod} = T @ (a_root \kron a_root)
    
    where a_root are coeffs of polynomial P (order lmax_root)
    and c_fod are coeffs of P^2 (order lmax_fod).
    Typically lmax_fod = 2 * lmax_root.
    
    Args:
        lmax_root: Max order of the 'square root' polynomial.
        lmax_fod: Max order of the target FOD (usually users specify this).
        
    Returns:
        Tensor G of shape (N_fod, N_root, N_root)
    """
    print(f"Angle 1: Precomputing Algebraic SOS Tensor (Root L={lmax_root} -> FOD L={lmax_fod})...")
    
    # 1. Define Basis
    basis_root, idx_root = get_real_sh_basis_sympy(lmax_root)
    basis_fod, idx_fod = get_real_sh_basis_sympy(lmax_fod)
    
    n_root = len(basis_root)
    n_fod = len(basis_fod)
    
    # 2. Compute Integrals
    # G[k, i, j] = Integral( Y_i * Y_j * Y_k ) dOmega
    # Because FOD = (sum a_i Y_i) * (sum a_j Y_j) = sum_{i,j} a_i a_j (Y_i Y_j)
    # We project (Y_i Y_j) onto basis Y_k:
    # coeff_k = Integral( (Y_i Y_j) * Y_k )  (Assuming orthonormal real basis)
    
    # Matrix G will be (N_fod, N_root, N_root)
    # This calculation can be slow in SymPy. 
    # Optimization: Use Znm orthogonality and selection rules if possible, or numerical quad if SymPy hangs.
    # SymPy 'integrate' on sphere is robust but slow.
    # We will use discrete integration on a high-resolution grid for speed (Algebraic-Numeric Hybrid).
    # This is "Agent 1's trick": Exact integration of polynomials <= N on the sphere can be done via Lebedev grids.
    
    # Or just standard numeric integration on dense grid.
    # Or, SymPy integration is fast enough for low L. L=8 -> N=45 coeffs.
    # 45*45*45 = 91125 integrals. Too slow for SymPy symbolic.
    
    # STRATEGY: Use pure JAX numerical integration on a high-order Lebedev grid.
    # It's exact for SH products up to grid degree.
    # P * P * P -> Order 3*L.
    # If L=8, need grid for L=24.
    # Simple and fast in JAX.
    pass

def precompute_sos_mapping(lmax_root: int = 4, lmax_fod: int = 8):
    """
    JAX-compatible generator for the SOS tensor.
    Uses numerical spherical integration (exact/high-precision).
    """
    # 1. Generate High-Res Grid (e.g. Fibonnaci or standard Sperical)
    # For exact integration of SH product of order L_total = 2*L_root + L_fod, 
    # we need sufficient points.
    # L_total = 4+4+8 = 16? No, max order of P is lmax_root. P^2 is 2*lmax_root.
    # We project P^2 onto Y_k (order lmax_fod).
    # Integral (P^2 * Y_k) has max order 3*max(L).
    
    # We use 'sh_basis_real' (Scipy-based) to support higher orders (L=8, etc.)
    # sh_basis_real_analytic is restricted to L=4.
    from dmipy_jax.utils.spherical_harmonics import sh_basis_real
    
    # Generate grid
    n_points = 4000
    key = jax.random.key(0)
    
    # Fibonacci sphere
    golden = (1 + 5**0.5)/2
    i = jnp.arange(n_points)
    phi = 2 * jnp.pi * (i / golden % 1)
    costheta = 1 - 2*(i + 0.5)/n_points
    theta = jnp.arccos(costheta)
    
    # Weights for Fibonacci: approx 4pi/N
    weights = jnp.full((n_points,), 4 * jnp.pi / n_points)
    
    # 2. Evaluate Basis matrices
    Y_root = sh_basis_real(theta, phi, lmax=lmax_root)
    Y_fod = sh_basis_real(theta, phi, lmax=lmax_fod)
    
    return Y_root, Y_fod, weights

@jax.jit
def build_sos_tensor(Y_root, Y_fod, weights):
    """
    Constructs the Triple Product Tensor.
    """
    # G_kij = sum_p w_p * Y_fod[p,k] * Y_root[p,i] * Y_root[p,j]
    G = jnp.einsum('p, pk, pi, pj -> kij', weights, Y_fod, Y_root, Y_root)
    return G

