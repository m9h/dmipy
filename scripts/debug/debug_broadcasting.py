
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

def debug_run():
    # shapes
    N_meas = 200
    N_roots = 20
    bvecs = jnp.ones((N_meas, 3))
    bvals = jnp.ones((N_meas,))
    mu = jnp.array([1.0, 0.0, 0.0])
    lambda_par = 1.0
    diameter = 1.0
    D_perp = 1.0
    tau = jnp.ones((N_meas, 1)) # expanded
    
    alpha = jnp.ones((1, N_roots))
    
    # Logic from c3_cylinder_callaghan
    
    # 1. Dot
    dot_prod = jnp.dot(bvecs, mu)
    print(f"dot: {dot_prod.shape}")
    
    q_mag = bvals # dummy
    
    sin_sq = 1 - dot_prod**2
    q_perp = q_mag * jnp.sqrt(sin_sq)
    print(f"q_perp: {q_perp.shape}")
    
    radius = diameter / 2
    q_argument = 2 * jnp.pi * q_perp * radius
    print(f"q_argument: {q_argument.shape}")
    q_arg_2 = q_argument**2
    q_arg_2_expanded = q_arg_2[:, None]
    print(f"q_arg_2_expanded: {q_arg_2_expanded.shape}")
    
    # m loop
    m = 1
    alpha_m = alpha[0, :] # (K,) ? No alpha is (1, K)
    alpha_m = alpha[:, m] # (1,) if alpha is (N, M). 
    # In model: alpha is (Roots, Functions)
    # alpha[k, n]. But in my code: alpha[:, m] -> (K,)
    # Let's match code:
    # alpha = np.zeros((n_roots, n_functions))
    alpha_arr = jnp.ones((20, 50))
    alpha_m = alpha_arr[:, m] # (20,)
    alpha_m_sq = alpha_m[None, :]**2 # (1, 20)
    
    print(f"alpha_m_sq: {alpha_m_sq.shape}")
    
    # J_prime
    # bessel_jn(z, v=v)
    J_val = 0.5 * (jsp.bessel_jn(q_argument, v=m-1) - jsp.bessel_jn(q_argument, v=m+1))
    print(f"J_val: {J_val.shape}")
    
    q_arg_J = (q_argument * J_val)**2
    q_arg_J_expanded = q_arg_J[:, None]
    print(f"q_arg_J_expanded: {q_arg_J_expanded.shape}")
    
    # Exp term
    # exp(-alpha^2 * D * tau / R^2)
    # tau (200, 1). alpha^2 (1, 20).
    exp_term_m = jnp.exp(-alpha_m_sq * D_perp * tau / radius**2)
    print(f"exp_term_m: {exp_term_m.shape}")
    
    numerator_factor = alpha_m_sq / (alpha_m_sq - m**2)
    print(f"numerator_factor: {numerator_factor.shape}")
    
    denom_m = (q_arg_2_expanded - alpha_m_sq)**2
    print(f"denom_m: {denom_m.shape}")
    
    # The failing line
    term_m = (8 * exp_term_m * numerator_factor * q_arg_J_expanded / denom_m)
    print(f"term_m: {term_m.shape}")
    
    print("Success!")

if __name__ == "__main__":
    debug_run()
