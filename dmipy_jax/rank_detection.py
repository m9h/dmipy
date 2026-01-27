
import jax
import jax.numpy as jnp
from dmipy_jax.basis.spherical_harmonics import real_sph_harm

def get_monomial_basis_matrix(max_order, num_points=100):
    """
    Compute the transformation matrix from Monomial coefficients to SH evaluations.
    Wait, we need SH -> Monomials.
    We equate S(u) = Sum c_lm Y_lm = Sum t_alpha u^alpha.
    We solve for t_alpha.
    
    We generate random points u_i.
    B_SH = [Y_lm(u_i)]
    B_Mon = [u_i^alpha]
    
    B_Mon @ t = B_SH @ c
    t = pinv(B_Mon) @ B_SH @ c
    
    This matrix M = pinv(B_Mon) @ B_SH maps c_lm to t_alpha.
    
    Args:
        max_order: int, e.g. 4.
        num_points: int, number of points for least squares fit.
    
    Returns:
        M: (N_monomials, N_sh) matrix.
    """
    key = jax.random.PRNGKey(42)
    # Generate random points on sphere
    z = jax.random.uniform(key, (num_points,), minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(key, (num_points,), minval=0.0, maxval=2*jnp.pi)
    theta = jnp.arccos(z)
    
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z_coords = jnp.cos(theta) # z is already cos(theta)
    
    points = jnp.stack([x, y, z_coords], axis=1) # (N, 3)
    
    # 1. SH Basis Matrix
    basis_cols = []
    # Ordering must match input SH coeffs. Assuming standard: l=0, then l=2(m=-2..2), l=4...
    # The existing invariants.py implies l=0, 2, 4.
    for l in range(0, max_order + 1, 2):
        for m in range(-l, l + 1):
            # real_sph_harm returns (N,)
            ylm = real_sph_harm(l, m, theta, phi)
            basis_cols.append(ylm)
    B_SH = jnp.stack(basis_cols, axis=1) # (N, N_sh)
    
    # 2. Monomial Basis Matrix
    # We need to define an ordering for monomials.
    # We want to match the Catalecticant indexing eventually.
    # Degree 4 monomials.
    # Let's list exponents (nx, ny, nz) such that sum=L.
    # Standard choice: Lexicographic or Reverse Lexicographic.
    # e.g. x^4, x^3y, x^3z, x^2y^2, ...
    # We will generate them and store the exponents mapping.
    
    exponents = get_monomial_exponents(max_order)
    
    mon_cols = []
    for (nx, ny, nz) in exponents:
        val = (points[:, 0]**nx) * (points[:, 1]**ny) * (points[:, 2]**nz)
        mon_cols.append(val)
    B_Mon = jnp.stack(mon_cols, axis=1) # (N, N_mon)
    
    # Solve M
    # t = M c
    # B_Mon t = B_SH c => t = pinv(B_Mon) B_SH c
    M = jnp.linalg.pinv(B_Mon) @ B_SH
    
    return M, exponents

def get_monomial_exponents(order):
    """
    Generate exponents (nx, ny, nz) sum to order.
    Sorted purely lexicographically implies x dominates, then y?
    We use standard iteration:
    x from L down to 0
      y from L-x down to 0
        z = L - x - y
    """
    exps = []
    for i in range(order, -1, -1): # x
        for j in range(order - i, -1, -1): # y
            k = order - i - j # z
            exps.append((i, j, k))
    return exps

def get_catalecticant_indices(L):
    """
    Generate pairs of indices (row_idx, col_idx) for the Catalecticant matrix C.
    C is square of size N_mon(L/2).
    Rows indexed by monomials of degree L/2.
    Cols indexed by monomials of degree L/2.
    Entry (u, v) corresponds to coefficient of monomial (u*v) in the tensor.
    
    Returns:
        row_exps: list of exponents for rows
        col_exps: list of exponents for cols
        map_indices: (Size, Size) array of integers. 
                     Each entry points to the index in the full Monomial vector (degree L)
                     corresponding to row_exp + col_exp.
    """
    half_L = L // 2
    row_exps = get_monomial_exponents(half_L)
    col_exps = row_exps # Symmetric
    
    full_exps = get_monomial_exponents(L)
    # Map (ex, ey, ez) -> index
    full_map = {e: i for i, e in enumerate(full_exps)}
    
    size = len(row_exps)
    indices = np_zeros((size, size), dtype=int)
    
    for r, r_exp in enumerate(row_exps):
        for c, c_exp in enumerate(col_exps):
            sum_exp = (r_exp[0]+c_exp[0], r_exp[1]+c_exp[1], r_exp[2]+c_exp[2])
            indices[r, c] = full_map[sum_exp]
            
    return indices

# We use a persistent cache or constants for these maps to avoid recomputing.
# Since we are in JAX, we can define them as globals or inside a closure.
# For simplicity, we assume L=4 is the main use case.

# Precompute helpers for L=4
import numpy as np

def np_zeros(shape, dtype): return np.zeros(shape, dtype=dtype)

_L = 4
_Indices_L4 = get_catalecticant_indices(_L)
# We can't easily precompute M without running JAX at import time which is bad?
# Alternatively, we assume M is computed once.
# We will compute M inside the JITted function or pass it.
# Actually, random number generation in `get_monomial_basis_matrix` makes it stochastic.
# We should fix the seed or implementation.
# For robustness, we'll implement `catalecticant_matrix` to take M as optional or compute it lazily.
# Better: hardcode the logic or use a deterministic grid (e.g. 6-designs).
# We'll use the stochastic approach but seeded for now inside the standard jax function? No, excessive.
# We will compute `M_global` at module level if possible, or create a factory.
# Let's put it in a setup function.

# Global M for L=4
# We will use a lazy initialization pattern or just compute it on first call?
# JAX doesn't like side effects.
# We will provide a function `get_transform_matrix()` and user passes it, 
# or we embed it as a constant.
# Let's assume we compute it on the fly (it's small, 15x15 inversion).
# We can wrap it in `jax.jit`.

@jax.jit
def _compute_M_L4():
    return get_monomial_basis_matrix(4, num_points=200)[0]

def construct_catalecticant(sh_coeffs, M=None):
    """
    Construct the Catalecticant matrix from SH coefficients.
    
    Args:
        sh_coeffs: (..., N_sh) array. For L=4, N_sh=15 (1+5+9).
        M: (Optional) Transformation matrix (N_mon, N_sh). 
           If None, computed on the fly (cached by JIT if constant?).
           
    Returns:
        C: (..., 6, 6) Catalecticant matrix (for L=4).
    """
    if M is None:
        M = _compute_M_L4()

    # Apply scaling to map generic SH coeffs (e.g. ODF/signal) to Order-4 Tensor coefficients
    # Factors derived from expansion of (u.d)^4 into Legendre Polynomials P0, P2, P4.
    # P0: 1/5, P2: 4/7, P4: 8/35
    # indices: 0 (L=0), 1-5 (L=2), 6-14 (L=4)
    # Correct weights accounting for SH normalization:
    # L=0: 1/5
    # L=2: 4/35
    # L=4: 8/315
    w0 = 1.0/5.0
    w2 = 4.0/35.0
    w4 = 8.0/315.0
    
    # Create mask (1, 15) or (15,)
    # Assumes last dim is 15
    scaling = jnp.concatenate([
        jnp.array([w0]),
        jnp.full((5,), w2),
        jnp.full((9,), w4)
    ])
    
    # sh_coeffs: (..., 15)
    coeffs_tensor = sh_coeffs * scaling
        
    # 1. Convert to Monomial Coefficients
    # shape: (..., N_mon)
    # These are coefficients c_alpha in P(x) = sum c_alpha x^alpha
    t_coeffs = jnp.dot(coeffs_tensor, M.T)
    
    # APPLY INVERSE MULTINOMIAL SCALING
    # c_alpha = (L choose alpha) * w_i * d_i^alpha
    # We want T_alpha = c_alpha / (L choose alpha) = w_i * d_i^alpha
    # Then C_{uv} = T_{u+v} = w_i * d_i^{u+v} = w_i * d_i^u * d_i^v
    # This matrix is strictly Rank N.
    
    # We need to compute multinomial coeffs for all monomials of degree 4.
    # Order matches get_monomial_exponents(4)
    exps = get_monomial_exponents(4)
    # Factorial L / (nx! ny! nz!)
    from jax.scipy.special import gammaln
    # exp(gammaln(n+1)) is factorial(n)
    
    def log_factorial(n):
        return gammaln(n + 1)
        
    def compute_multinomial(exponents, L):
        # exponents: (N, 3)
        # res = L! / (e1! e2! e3!)
        log_num = log_factorial(L)
        log_den = log_factorial(exponents[:, 0]) + log_factorial(exponents[:, 1]) + log_factorial(exponents[:, 2])
        return jnp.exp(log_num - log_den)

    # Convert exps to array
    exps_arr = jnp.array(exps)
    multinomials = compute_multinomial(exps_arr, 4)
    
    # Scale t_coeffs
    # t_coeffs: (..., 15)
    # multinomials: (15,)
    t_coeffs_scaled = t_coeffs / multinomials
    
    # 2. Fill Matrix
    # We use the indices map.
    # _Indices_L4 is numpy array, we convert to jax array or use take.
    inds = jnp.array(_Indices_L4)
    
    # C: (..., 6, 6)
    # We use fancy indexing on the SCALED coefficients
    C = t_coeffs_scaled[..., inds]
    
    return C

def rank_determination(catalecticant, threshold=1e-4):
    """
    Determine rank based on Singular Values of Catalecticant.
    
    Args:
        catalecticant: (..., 6, 6)
        threshold: float, relative threshold for singular values.
        
    Returns:
        rank: (...,) integer {1, 2, 3} based on N fibers.
              Actually returns algebraic rank. Can be > 3 if isotropic?
              We clamp to 1..3 for this specific task.
    """
    S = jnp.linalg.svd(catalecticant, compute_uv=False)
    # S is sorted descending.
    # Normalize by max singular value (S[0])
    S_norm = S / (S[..., :1] + 1e-12)
    
    # Count how many are above threshold
    # But usually:
    # Rank 1: s1 >> s2
    # Rank 2: s1, s2 >> s3
    # Rank 3: s1, s2, s3 >> s4
    
    # We can count > threshold.
    r = jnp.sum(S_norm > threshold, axis=-1)
    
    # For dMRI usually we look for up to 3 fibers.
    # We can clamp.
    return jnp.clip(r, 1, 3)

def waring_decomposition_rank1(catalecticant):
    """
    Extract fiber direction for Rank 1.
    For Rank 1, C = lambda * v * v.T
    The monomial vector v (degree 2) corresponds to (d)^2.
    d = (x, y, z).
    v = (x^2, xy, xz, y^2, yz, z^2) if lex order?
    Actually v is the vector of evaluations of basis on d.
    If we find v (principal eigenvector of C), we can extract (x, y, z).
    
    Args:
        catalecticant: (..., 6, 6)
    
    Returns:
        direction: (..., 3) (normalized)
    """
    # Principal eigenvector
    U, S, Vt = jnp.linalg.svd(catalecticant)
    v = U[..., 0] * jnp.sqrt(S[..., 0:1]) # Scale?
    
    # v corresponds to [x^2, xy, xz, y^2, yz, z^2]
    # We can extract x, y, z from xy, xz, x^2 etc.
    # Let's say v = [v0, v1, v2, v3, v4, v5]
    # x = sqrt(v0) * sign(...) ?
    # Better: x*x = v0, x*y = v1 -> y = v1/x.
    # To be robust against x=0:
    # direction parallel to (sqrt(v0), sign(v1)*sqrt(v3), sign(v2)*sqrt(v5))?
    # Actually signs are ambiguous in v.
    # But v1 = xy, v2 = xz, v4 = yz.
    # x^2=v0, y^2=v3, z^2=v5.
    
    # Recover magnitude
    # x = sqrt(|v0|)
    # y = sqrt(|v3|) * sign(v1) * sign(x)? 
    #   if x>0: sign(y) = sign(v1).
    #   if x<0: sign(y) = -sign(v1).
    # Overall sign of d doesn't matter. Assume x > 0.
    
    abs_d = jnp.sqrt(jnp.abs(jnp.stack([v[..., 0], v[..., 3], v[..., 5]], axis=-1)))
    
    # Signs relative to x (assume x+)
    # y_sign = sign(v1)
    # z_sign = sign(v2)
    # check consistency with v4 (yz)?
    # sign(v4) should match sign(v1)*sign(v2).
    
    s_x = jnp.ones_like(abs_d[..., 0])
    s_y = jnp.sign(v[..., 1])
    s_z = jnp.sign(v[..., 2])
    
    # If v0 (x^2) is very small, we might be looking at y-z plane.
    # We need a robust extraction.
    # Standard approach: Take column with largest diagonal entry.
    # M = [[x^2 xy xz], [xy y^2 yz], [xz yz z^2]]
    # This matrix is part of C? 
    # C rows: x^2, xy, xz, ...
    # Wait, C is 6x6.
    # A Rank 1 tensor T = (d)^4.
    # C_ij = d^(row_exp + col_exp).
    # row 0 (x^2), col 0 (x^2) -> x^4.
    # The top-left 3x3 block of C (rows x^2,xy,xz; cols x^2,xy,xz) ?
    # Not quite.
    # Just constructing the 3x3 matrix D = d * d.T from the eigenvector v?
    # v corresponds to monomials of degree 2: x^2, xy, xz, ...
    # So v IS the flattened D matrix (upper triangular part).
    
    # Exponents for N=2 (L/2):
    # (2,0,0) -> x^2
    # (1,1,0) -> xy
    # (1,0,1) -> xz
    # (0,2,0) -> y^2
    # (0,1,1) -> yz
    # (0,0,2) -> z^2
    
    # v components: 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz.
    # Construct 3x3 matrix:
    # [[v0, v1, v2],
    #  [v1, v3, v4],
    #  [v2, v4, v5]]
    
    # This matrix is proportional to d * d.T
    # We compute its eigenvector corresponding to max eigenvalue.
    xx = v[..., 0]
    xy = v[..., 1]
    xz = v[..., 2]
    yy = v[..., 3]
    yz = v[..., 4]
    zz = v[..., 5]
    
    row1 = jnp.stack([xx, xy, xz], axis=-1)
    row2 = jnp.stack([xy, yy, yz], axis=-1)
    row3 = jnp.stack([xz, yz, zz], axis=-1)
    
    mat = jnp.stack([row1, row2, row3], axis=-2) # (..., 3, 3)
    
    # Eigen decomp of 3x3 real symmetric matrix
    vals, vecs = jnp.linalg.eigh(mat)
    # Last one is largest
    d = vecs[..., :, -1]
    
    return d


def initialize_tt_rank(rank):
    """
    Map algebraic rank to TT-rank.
    For sum of N fibers (rank N), the TT-ranks of the tensor are small.
    Usually (1, N, N, ..., 1) or bounded by N.
    """
    return rank

# Note: Waring decomposition for N=2, 3 requires polynomial root finding.
# We include placeholders for now as N=1 is the most critical for initialization
# and N>1 often handled by optimization.
# Implementing full analytical Waring for N=2,3 in JAX is complex 
# (requires solving cubic equations / eigen of companion).
# We can provide the N=1 solution and leave TODO or use general optimization if needed.
# The user asked for "Algebraic recipe".
# Recipe for N=2:
# 1. Kernel of C has dimension 6-2=4.
# 2. Pick 2 quadrics Q1, Q2 from kernel.
# 3. Restrict to Plane? Or solve Q1=Q2=0.
# 4. This is efficiently done by finding generalized eigenvalues of the pencil.

