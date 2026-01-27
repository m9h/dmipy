import jax
import jax.numpy as jnp
from dmipy_jax.core.roots import differentiable_roots

def solve_microstructure(invariants, delta_b=1000.0):
    """
    Direct Inversion of Microstructure parameters using Differentiable Prony's Method.
    
    Assumes `invariants` (signal inputs) correspond to 4 equidistant b-value measurements
    (or moments equivalent to them) for a 2-compartment model.
    Model: S_k = w_1 * z_1^k + w_2 * z_2^k, where z_i = exp(-delta_b * D_i).
    
    Args:
        invariants: (..., 4) Signal values [S0, S1, S2, S3].
        delta_b: Step size of b-values (difference between consecutive measurements).
        
    Returns:
        params: (..., 3) [D_1, D_2, f_1].
                D_1, D_2 are diffusivities. f_1 is volume fraction of first compartment.
                (f_2 is implicitly 1 - f_1 if normalized, or derived from weights).
    """
    # 1. Construct Polynomial Coefficients (Prony's Method for K=2)
    # Solve linear system H * c = y
    # [S0 S1] [c2] = [-S2]
    # [S1 S2] [c1]   [-S3]
    
    y0, y1, y2, y3 = invariants[..., 0], invariants[..., 1], invariants[..., 2], invariants[..., 3]
    
    # Construct H and rhs
    # row1: c2*S0 + c1*S1 = -S2
    # row2: c2*S1 + c1*S2 = -S3
    
    # Matrix A = [[S1, S0], [S2, S1]] (for [c1, c2])
    # RHS b = [-S2, -S3]
    
    # Determinant of A
    det = y1 * y1 - y2 * y0
    
    # Solve using Cramer's rule or explicit inverse for 2x2
    # inv(A) = 1/det * [[S1, -S0], [-S2, S1]]
    # [c1] = 1/det * ( S1*(-S2) - S0*(-S3) ) = (S0 S3 - S1 S2) / (S1^2 - S0 S2)
    # [c2] = 1/det * (-S2*(-S2) + S1*(-S3)) = (S2^2 - S1 S3) / (S1^2 - S0 S2)
    
    # Make robust to small determinant?
    # det needs to be non-zero. If zero, singular (likely single compartment).
    safe_det = jnp.where(jnp.abs(det) < 1e-9, 1e-9, det)
    
    c1 = (y0 * y3 - y1 * y2) / safe_det
    c2 = (y2 * y2 - y1 * y3) / safe_det # Wait, matrix above: inverse had -S2?
    
    # Re-check inverse:
    # A = [[a, b], [c, d]] -> inv = 1/(ad-bc) [[d, -b], [-c, a]]
    # A = [[y1, y0], [y2, y1]] (Note order c1, c2)
    # det = y1*y1 - y0*y2
    # inv = [[y1, -y0], [-y2, y1]] / det
    # rhs = [-y2, -y3]
    # c1 = (y1*(-y2) - y0*(-y3))/det = (y0 y3 - y1 y2) / det. Correct.
    # c2 = (-y2*(-y2) + y1*(-y3))/det = (y2 y2 - y1 y3) / det. Correct.
    
    # Coefficients for poly: z^2 + c1 z + c2 = 0
    # coeffs array: [1, c1, c2]
    
    # We need to stack for differentiable_roots which expects (N+1,)
    # But we want to vmap this kernel.
    # differentiable_roots takes a single 1D array.
    # We'll need to define a helper that takes 1D coeffs and map it.
    
    # Let's organize the vmap at the end or assume scalar inputs logic here that gets vmapped.
    # coeffs shape (3,)
    coeffs = jnp.stack([jnp.ones_like(c1), c1, c2], axis=-1)
    
    # 2. Find Roots
    # We call differentiable_roots. 
    # Since differentiable_roots is designed for 1D input, we wrap it or use vmap inside?
    # Better: The caller (this function) usually processes a single voxel if used inside vmap,
    # OR we explicitly vmap the root finder.
    # Let's assume this function handles batches.
    # But differentiable_roots (custom_vjp) signature expects `coeffs` (1D).
    # We must vmap the root finder to apply to batch.
    roots_vjp_vmap = jax.vmap(differentiable_roots)
    
    # shape (..., 2)
    roots = roots_vjp_vmap(coeffs) 
    
    # 3. Post-Processing / Selection
    
    # Roots z = exp(-b * D)
    # D = -ln(z) / delta_b
    
    # Filter:
    # We want real, positive roots.
    # Complex roots occur due to noise. 
    # We can take the real part, or magnitude?
    # Magnitude is robust for oscillatory noise in z.
    z_mag = jnp.abs(roots)
    
    # Ensure z is in (0, 1] range ideally (decay).
    z_clamped = jnp.clip(z_mag, 1e-6, 1.0)
    
    Ds = -jnp.log(z_clamped) / delta_b
    
    # Select biologically valid
    # Sort or identify compartments?
    # Usually we sort by diffusivity (slow, fast).
    # axis -1 is roots.
    Ds_sorted = jnp.sort(Ds, axis=-1) # [Small, Large] -> [Slow, Fast] (usually restricted vs free)
    
    # Recover Fractions (Weights)
    # Linear system for w1, w2:
    # S0 = w1 + w2
    # S1 = w1 z1 + w2 z2
    # ...
    # Easier: Just use S0 and S1 (or all 4 via least squares).
    # Vandermonde matrix V for the solved roots.
    # M = [[1, 1], [z1, z2]]
    # b = [S0, S1]
    # w = inv(M) * b
    
    z_sorted = jnp.exp(-Ds_sorted * delta_b)
    z1, z2 = z_sorted[..., 0], z_sorted[..., 1]
    
    # Determinant: z2 - z1
    denom = z2 - z1
    # Check for degeneracy (z1 ~ z2)
    safe_denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)
    
    # inv(M) = [[z2, -1], [-z1, 1]] / denom
    # w1 = (z2 * S0 - S1) / denom
    # w2 = (-z1 * S0 + S1) / denom
    
    w1 = (z2 * y0 - y1) / safe_denom
    w2 = (-z1 * y0 + y1) / safe_denom
    
    # Calculate fraction f1 = w1 / (w1 + w2)
    # Ideally w1+w2 = S0.
    total_w = w1 + w2
    f1 = w1 / jnp.where(total_w == 0, 1.0, total_w)
    
    # Clip f1
    f1 = jnp.clip(f1, 0.0, 1.0)
    
    # Return [D1, D2, f1]
    # (D1 is slow, D2 is fast)
    return jnp.stack([Ds_sorted[..., 0], Ds_sorted[..., 1], f1], axis=-1)

