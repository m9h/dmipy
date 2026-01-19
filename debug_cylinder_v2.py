
import jax
import jax.numpy as jnp
from dmipy_jax.signal_models import cylinder_models
import sys

def log(msg):
    with open("debug_cylinder_v2.log", "a") as f:
        f.write(msg + "\n")

def test_cylinder_standalone():
    log("Starting debug...")
    try:
        cylinder = cylinder_models.RestrictedCylinder()
        
        N = 5
        bvals = jnp.ones(N) * 1000.0
        bvecs = jnp.zeros((N, 3))
        bvecs = bvecs.at[:, 0].set(1.0)
        
        params = {
            'lambda_par': 1.7e-9,
            'mu': jnp.array([jnp.pi/2, 0.0]),
            'diameter': 5e-6,
            'big_delta': 0.03,
            'small_delta': 0.01
        }
        
        # Explicitly jit to ensure tracing happens if prints are inside jit-ted code in module
        jitted_cyl = jax.jit(cylinder)
        res = jitted_cyl(bvals, bvecs, **params)
        log(f"Standalone result shape for N={N}: {res.shape}")
        
        # Helper manual calc
        mu_in = params['mu']
        theta = mu_in[0]
        phi = mu_in[1]
        st, ct = jnp.sin(theta), jnp.cos(theta)
        sp, cp = jnp.sin(phi), jnp.cos(phi)
        mu_cart = jnp.array([st*cp, st*sp, ct])
        log(f"Manual mu_cart shape: {mu_cart.shape}")
        
        res_direct = cylinder_models.c2_cylinder(
            bvals, bvecs, mu_cart, 
            params['lambda_par'], params['diameter'], 
            params['big_delta'], params['small_delta']
        )
        log(f"Direct c2_cylinder shape: {res_direct.shape}")
        
        # Flattened logic
        log("--- Flattened c2_cylinder execution ---")
        try:
            mu = mu_cart
            lambda_par = params['lambda_par']
            diameter = params['diameter']
            big_delta = params['big_delta']
            small_delta = params['small_delta']
            
            # 1. Parallel Signal
            log(f"Input shapes: bvecs {bvecs.shape}, mu {mu.shape}")
            dot_prod = jnp.dot(bvecs, mu)
            log(f"dot_prod shape: {dot_prod.shape}")
            
            signal_par = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
            log(f"signal_par shape: {signal_par.shape}")
            
            # 2. Perpendicular Signal
            tau = big_delta - small_delta / 3.0
            log(f"tau: {tau}, q_mag input len {len(bvals)}")
            q_mag = jnp.sqrt(bvals / (tau + 1e-9)) / (2 * jnp.pi)
            log(f"q_mag shape: {q_mag.shape}")
            
            sin_theta_sq = 1 - dot_prod**2
            sin_theta_sq = jnp.clip(sin_theta_sq, 0.0, 1.0)
            log(f"sin_theta_sq shape: {sin_theta_sq.shape}")
            
            q_perp = q_mag * jnp.sqrt(sin_theta_sq)
            log(f"q_perp shape: {q_perp.shape}")
            
            radius = diameter / 2.0
            argument = 2 * jnp.pi * q_perp * radius
            log(f"argument shape: {argument.shape}")
            
            valid_mask = argument > 1e-6
            safe_arg = jnp.where(valid_mask, argument, 1.0)
            log(f"safe_arg shape: {safe_arg.shape}")
            
            import jax.scipy.special as jsp
            j1_term = 2 * jsp.bessel_jn(safe_arg, v=1) / safe_arg
            log(f"j1_term shape: {j1_term.shape}")
            
            signal_perp = j1_term ** 2
            signal_perp = jnp.where(valid_mask, signal_perp, 1.0)
            log(f"signal_perp shape: {signal_perp.shape}")
            
            total_signal = signal_par * signal_perp
            log(f"Total signal shape: {total_signal.shape}")
            
        except Exception as e:
            log(f"Flattened execution failed: {e}")
            import traceback
            traceback.print_exc(file=open("debug_cylinder_v2.log", "a"))
        N=2
        bvals = jnp.ones(N) * 1000.0
        bvecs = jnp.zeros((N, 3))
        bvecs = bvecs.at[:, 0].set(1.0)
        res = cylinder(bvals, bvecs, **params)
        print(f"Standalone result shape for N={N}: {res.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cylinder_standalone()
