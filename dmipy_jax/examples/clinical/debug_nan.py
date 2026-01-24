
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.signal_models.cylinder_models import CallaghanRestrictedCylinder
from dmipy_jax.distributions.distributions import DD1Gamma

def debug():
    print("Debugging components...")
    
    # 1. Check DD1Gamma
    alpha = 4.0
    beta = 0.5e-6
    dist = DD1Gamma(alpha=alpha, beta=beta)
    try:
        radii, pdf = dist()
        print(f"DD1Gamma: radii range [{jnp.min(radii)}, {jnp.max(radii)}], PDF sum {jnp.sum(pdf)}")
        if jnp.isnan(radii).any() or jnp.isnan(pdf).any():
            print("NaN in DD1Gamma!")
    except Exception as e:
        print(f"DD1Gamma failed: {e}")

    # 2. Check CallaghanRestrictedCylinder at single point
    cyl = CallaghanRestrictedCylinder()
    
    # Params
    mu = jnp.array([1.57, 0.0])
    lambda_par = 1.7e-9
    diameter = 2.0e-6
    diffusion_perp = 1.7e-9
    
    # Protocol
    bval = 3000.0 # s/mm^2
    bvec = jnp.array([1.0, 0.0, 0.0]) # Parallel to x-axis (and mu) -> perp diffusion irrelevant
    # Let's try perp direction
    bvec_perp = jnp.array([0.0, 1.0, 0.0])
    
    big_delta = 0.0252
    small_delta = 0.0152
    
    # Signal call
    try:
        # Case 1: Parallel
        s_par = cyl(
            jnp.array([bval]), 
            jnp.array([bvec]), 
            mu=mu, 
            lambda_par=lambda_par, 
            diameter=diameter, 
            diffusion_perpendicular=diffusion_perp, 
            big_delta=big_delta, 
            small_delta=small_delta
        )
        print(f"Callaghan (Parallel): {s_par}")
        
        # Case 2: Perpendicular
        s_perp = cyl(
            jnp.array([bval]), 
            jnp.array([bvec_perp]), 
            mu=mu, 
            lambda_par=lambda_par, 
            diameter=diameter, 
            diffusion_perpendicular=diffusion_perp, 
            big_delta=big_delta, 
            small_delta=small_delta
        )
        print(f"Callaghan (Perpendicular): {s_perp}")
        
    except Exception as e:
        print(f"Callaghan failed: {e}")
        
    # 3. Check specific term in Callaghan
    # q_mag calc
    tau = big_delta - small_delta/3.0
    q_mag = jnp.sqrt(bval / (tau + 1e-12)) / (2 * jnp.pi) * 1e3
    radius = diameter / 2.0
    q_perp = q_mag * 1.0 # assume perp
    q_argument = 2 * jnp.pi * q_perp * radius
    print(f"bval={bval}, tau={tau}, q_mag={q_mag}, q_argument={q_argument}")
    
    # Check exp term
    # exp(-alpha^2 * D * tau / R^2)
    # alpha[0,0] ~ 3.83 (first root of J1') (?) No, J1'=0 roots.
    # jnp_zeros(m, n_roots)
    # m=0: roots of J0'=0 -> J(-1) - J(1)? J0' = -J1. So roots of J1=0.
    # First root of J1 is 3.83.
    # alpha^2 ~ 14.
    # D * tau / R^2 = 1.7e-9 * 0.02 / (1e-6)^2 = 3.4e-11 / 1e-12 = 34.
    # exp(-14 * 34) = exp(-476) ~ 0.
    # Safe.
    
    # What if diameter is very small? 0.1e-6.
    diameter_small = 0.1e-6
    radius_small = diameter_small / 2.0
    # D * tau / R^2 = 1.7e-9 * 0.02 / (0.05e-6)^2 = 3.4e-11 / 2.5e-15 = 1.36e4
    # exp(-14 * 13600) = exp(-190000) = 0.
    # Safe.
    
    # What if diameter is large? 20um.
    diameter_large = 20e-6
    radius_large = 10e-6
    # D * tau / R^2 = 3.4e-11 / 1e-10 = 0.34
    # exp(-14 * 0.34) = exp(-4.7) ~ 0.009.
    # Safe.
    
    # DENOMINATOR
    # denom = (q_arg^2 - alpha^2)^2
    # Singularity if q_arg == alpha.
    # q_arg = 2 * pi * q * R.
    # q_mag ~ 1e5 m^-1?
    # q_mag = sqrt(3000 / 0.02) / 2pi * 1e3 = sqrt(150000)/6.28 * 1e3 = 387/6.28 * 1e3 = 61 * 1e3 = 6.1e4.
    # radius = 1e-6.
    # q_arg = 2 * 3.14 * 6.1e4 * 1e-6 = 0.38.
    # alpha ~ 3.83.
    # q_arg << alpha. No resonance generally.
    
    # Unless bval is massive?
    # bval = 30000? q_mag ~ 2e5. q_arg ~ 1.2. Still < 3.83.
    # So unlikely to hit resonance for small axons.
    
    # 4. Check Bessel behavior
    print("Checking Bessel...")
    try:
        from jax.scipy import special as jsp
        z = jnp.array([0.0003, 0.38, 1.0])
        j0 = jsp.bessel_jn(z, v=0)[0] # bessel_jn returns [J0, ...] usually? Wait, checking doc/behavior.
        # My bessel_jn_fixed says: vals = jsp.bessel_jn(z, v=v_int); return vals[v_int]
        # jsp.bessel_jn(z, v=n) returns array of length n+1.
        
        vals_0 = jsp.bessel_jn(z, v=0)
        print(f"J0(z): {vals_0}")
        
        vals_1 = jsp.bessel_jn(z, v=1)
        print(f"J0..J1(z): {vals_1}")
        
        vals_50 = jsp.bessel_jn(z, v=50)
        print(f"J50(z) shape: {vals_50.shape}")
        if jnp.isnan(vals_50).any():
            print("NaN in J50!")
    except Exception as e:
        print(f"Bessel check failed: {e}")

    print("Debug check finished.")

if __name__ == "__main__":
    debug()
