
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

def j1_taylor(z, n_terms=20):
    # Taylor series approximation for J1(z) using JAX scan for stability.
    # Safe for small z (e.g. z < 4).
    
    def series_sum(z_val):
        z2_4 = z_val**2 / 4.0
        
        def body(carry, k):
            current_sum, current_term = carry
            # term[k] = term[k-1] * (-1) * (z^2/4) / (k * (k+1))
            # k is 1-based index here corresponding to term order?
            # series: J1(z) = (z/2) sum_{k=0} (-1)^k (z/2)^2k / (k! (k+1)!)
            # k=0 term: z/2.
            # k=1 term: -(z/2) * (z^2/4) / 2. = previous * (-z^2/4) / (1 * 2).
            # k=2 term: (z/2) * (z^2/4)^2 / 12. = previous * (-z^2/4) / (2 * 3).
            # So multiplier is -z2_4 / (k * (k+1)) where k is the index of the NEW term.
            
            mult = -z2_4 / (k * (k+1))
            next_term = current_term * mult
            new_sum = current_sum + next_term
            return (new_sum, next_term), None
            
        term_0 = z_val / 2.0
        init = (term_0, term_0)
        
        # Scan k from 1 to n_terms-1
        ks = jnp.arange(1, n_terms, dtype=jnp.float32)
        (final_sum, _), _ = jax.lax.scan(body, init, ks)
        return final_sum

    return jax.vmap(series_sum)(jnp.atleast_1d(z))

def run():
    print("Verifying J1 Taylor Approximation (Corrected)")
    # Test range including small z
    z_vals = np.linspace(0.0, 5.0, 50)
    
    jax_j1 = j1_taylor(z_vals)
    scipy_j1 = special.j1(z_vals)
    
    err = np.abs(jax_j1 - scipy_j1)
    # Check limit 4.0
    max_err_4 = np.max(err[z_vals <= 4.0])
    print(f"Max Error (z <= 4.0): {max_err_4:.2e}")
    
    # Specific points
    print(f"J1(0): {jax_j1[0]}")
    print(f"J1(4.0): {jax_j1[np.argmin(np.abs(z_vals-4.0))]}")
    
    if max_err_4 < 1e-5:
        print("SUCCESS: Approximation is accurate.")
    else:
        print("FAILURE: Error too high.")

if __name__ == "__main__":
    run()
