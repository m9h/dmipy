import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def differentiable_roots(coeffs, strip_leading_zeros=True):
    """
    Computes roots of a polynomial with coefficients `coeffs`.
    Differentiable via Implicit Function Theorem.
    
    Args:
        coeffs: (N+1,) Coefficients of polynomial p(x) = c[0] * x^N + ... + c[N].
                (Standard numpy/JAX ordering: highest degree first).
    
    Returns:
        roots: (N,) complex roots.
    """
    # jnp.roots expects rank-1 array (assumes scalar coefficients in one poly).
    # We call standard jax.numpy.roots (which delegates to lapack/eigvals).
    # Note: jnp.roots gradients are not implemented by default in JAX usually, 
    # or involve eigendecomposition gradients which can be unstable for Companion matrices.
    # We provide a cleaner IFT gradient.
    return jnp.roots(coeffs)

def differentiable_roots_fwd(coeffs, strip_leading_zeros=True):
    roots = differentiable_roots(coeffs, strip_leading_zeros)
    return roots, (roots, coeffs)

def differentiable_roots_bwd(strip_leading_zeros, res, g_roots):
    """
    Backward pass for roots.
    
    P(z, c) = sum_{k=0}^N c_{N-k} * z^k = 0
    (Note: python convention c[0] is high order).
    
    Let P(z, c) = c[0]z^N + c[1]z^{N-1} + ... + c[N] = 0.
    
    Differential:
    dP/dz * dz + dP/dc * dc = 0
    => dz/dc = - (dP/dc) / (dP/dz)
    
    dP/dc_k is coefficient of c_k in P. 
    c_k multiplies z^{N-k}.
    So dP/dc_k = z^{N-k}.
    
    Therefore: dz / dc_k = - z^{N-k} / P'(z).
    
    Inputs:
        res: (roots, coeffs)
        g_roots: Gradient w.r.t roots (same shape as roots).
        
    Returns:
        g_coeffs: Gradient w.r.t coeffs.
    """
    z, c = res
    g_z = g_roots # shape (N,)
    
    # 1. Evaluate P'(z) at all roots.
    # P'(z) can be computed by evaluating poly with derivative of coefficients.
    # coeffs derivative: [N*c[0], (N-1)*c[1], ..., 1*c[N-1]]
    # (Constant term c[N] disappears).
    
    N = c.shape[0] - 1
    # Derivative coeffs:
    # c_deriv[k] = c[k] * (N - k)
    # Exclude last term
    poly_orders = jnp.arange(N, 0, -1) # N, N-1, ..., 1
    c_deriv = c[:-1] * poly_orders
    
    # Evaluate P'(z)
    P_prime_z = jnp.polyval(c_deriv, z)
    
    # Avoid division by zero (multiple roots).
    # In practice, multiple roots have undefined gradients or infinite sensitivities.
    # We add epsilon or handle safely? IFT fails at multiple roots.
    # For now, let it be (standard numerical behavior).
    # If P'(z) is tiny, gradient explodes.
    
    # dz_j / dc_k = - (z_j)^(N-k) / P'(z_j)
    
    # We want to compute g_coeffs[k] = sum_j (g_z[j] * dz_j/dc_k)
    # = sum_j ( g_z[j] * -1/P'(z_j) * (z_j)^(N-k) )
    
    # Let w_j = - g_z[j] / P'(z_j)
    w = - g_z / P_prime_z # Shape (N,)
    
    # We need to compute sum_j w_j * (z_j)^(N-k) for k=0..N
    # This is a Vandermonde-like structure.
    # Powers of z:
    # k=0 -> power N
    # k=1 -> power N-1
    # ...
    # k=N -> power 0
    
    # Powers matrix: V[k, j] = (z_j)^(N-k) ? No
    # We are summing over roots j.
    # Coeff grad index k corresponds to term z^{N-k}.
    
    # Let's vectorize.
    # powers: (N+1,) range N..0
    powers = jnp.arange(N, -1, -1) # [N, N-1, ..., 0]
    
    # z: (N_roots,) -> (N,) usually matching N
    # (N+1 coeffs -> N roots)
    
    # z_powers: (N_roots, N+1)
    # z_powers[j, k] = z_j ** powers[k]
    # We can broadcast.
    
    z_expanded = z[:, None] # (N_roots, 1)
    powers_expanded = powers[None, :] # (1, N+1)
    
    V = jnp.power(z_expanded, powers_expanded) # (N_roots, N+1)
    
    # g_c[k] = sum_j w_j * V[j, k]
    # g_c = w @ V
    
    g_coeffs = jnp.dot(w, V)
    
    # Note: jnp.roots solutions are generally complex.
    # If original coeffs are real, we expect g_coeffs to be real?
    # Not necessarily. If cost function uses complex roots, derivative is complex.
    # If coeffs are real and we strictly stay real, we might take real part.
    # But JAX handles complex differentiation appropriately (Holomorphic).
    # If the user's loss function effectively treats roots as just numbers, 
    # we return gradient as computed.
    
    return (g_coeffs,)

differentiable_roots.defvjp(differentiable_roots_fwd, differentiable_roots_bwd)
