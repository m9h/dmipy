import jax.numpy as jnp
from jax import jit
from dmipy_jax.constants import SPHERE_ROOTS, GYRO_MAGNETIC_RATIO

@jit
def g3_sphere(bvals, bvecs, diameter, diffusion_constant, big_delta, small_delta):
    """
    Computes signal for restricted diffusion in a Sphere (Soma) using
    the Gaussian Phase Distribution (GPD) approximation (Murday & Cotts, 1968).
    """
    radius = diameter / 2.0
    
    # 1. Gradients
    tau = big_delta - small_delta / 3.0
    G_mag = jnp.sqrt(bvals / (tau + 1e-9)) / (GYRO_MAGNETIC_RATIO * small_delta)
    
    # 2. Roots (Dimensionless mu)
    # alpha_sq_broad = mu^2
    alpha_sq = SPHERE_ROOTS ** 2
    alpha_sq_broad = alpha_sq[None, :] 
    
    # 3. Time Constants (Dm = D * mu^2 / R^2)
    # This part was correct
    Dm_alpha2 = diffusion_constant * alpha_sq_broad / (radius**2)
    
    # 4. The Denominator (The Fix)
    # We use dimensionless mu for the check (mu^2 - 2)
    # The physical 1/alpha^2 term contributes an R^2 factor to the numerator
    # because 1/alpha_phys^2 = R^2 / mu^2
    denom_dimensionless = alpha_sq_broad * (alpha_sq_broad - 2)
    
    # 5. Time Terms
    exp_1 = jnp.exp(-Dm_alpha2 * small_delta)
    exp_2 = jnp.exp(-Dm_alpha2 * big_delta)
    exp_3 = jnp.exp(-Dm_alpha2 * (big_delta - small_delta))
    exp_4 = jnp.exp(-Dm_alpha2 * (big_delta + small_delta))
    
    time_term = (
        2 * small_delta 
        - (2 + exp_3 - 2 * exp_2 - 2 * exp_1 + exp_4) / Dm_alpha2
    )
    
    # 6. Summation
    # We multiply by radius**2 because of the conversion from physical alpha to mu
    # We also need to divide by Dm_alpha2 ($D \alpha^2$) as per the formula
    sum_term = jnp.sum((time_term / denom_dimensionless / Dm_alpha2), axis=1) * (radius ** 2)
    
    # 7. Final Signal
    prefactor = 2 * (GYRO_MAGNETIC_RATIO * G_mag) ** 2
    log_E = -prefactor * sum_term
    
    return jnp.exp(log_E)