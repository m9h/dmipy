import jax
import jax.numpy as jnp
import scipy.special as ssp
from jax import pure_callback, jit
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO

def j1(x):
    # Callback to scipy.special.j1
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pure_callback(lambda z: ssp.j1(z).astype(z.dtype), result_shape, x)

@jit
def c2_cylinder(bvals, bvecs, mu, lambda_par, diameter, big_delta, small_delta):
    """
    Computes signal for a Cylinder with finite radius (Soderman approximation).
    
    Args:
        bvals: (N,) array of b-values in s/mm^2.
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (mm^2/s).
        diameter: Cylinder diameter in meters.
        big_delta: Diffusion time / pulse separation (s).
        small_delta: Pulse duration (s).
        
    Returns:
        (N,) array of signal attenuation (0.0 to 1.0).
    """
    # 1. Parallel Signal (Stick)
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    signal_par = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    
    # 2. Perpendicular Signal (Soderman / Stejskal-Tanner)
    # We need to calculate q_perp.
    # q = gamma * G * delta / (2*pi)
    # But usually we have b-values. b = (gamma * G * delta)^2 * (Delta - delta/3)
    # So q = sqrt(b / (Delta - delta/3)) / (2*pi)
    
    # However, standard dmipy uses q directly if available, or derives it.
    # Let's derive q_mag from bvals.
    tau = big_delta - small_delta / 3.0
    q_mag = jnp.sqrt(bvals / (tau + 1e-9)) / (2 * jnp.pi)
    
    # Project gradients perpendicular to fiber axis
    # |g_perp| = |g - (g . mu)mu| = |g| * sqrt(1 - (g_hat . mu)^2)
    # q_perp = q_mag * sqrt(1 - dot_prod^2)
    sin_theta_sq = 1 - dot_prod**2
    # Clip to avoid negative due to precision
    sin_theta_sq = jnp.clip(sin_theta_sq, 0.0, 1.0) 
    q_perp = q_mag * jnp.sqrt(sin_theta_sq)
    
    radius = diameter / 2.0
    argument = 2 * jnp.pi * q_perp * radius
    
    # E_perp = [ 2 * J1(2*pi*q*R) / (2*pi*q*R) ]^2
    # Handle singularity at argument=0 where J1(x)/x -> 0.5, so 2*0.5=1
    
    # Safe division
    valid_mask = argument > 1e-6
    safe_arg = jnp.where(valid_mask, argument, 1.0)
    
    j1_term = 2 * j1(safe_arg) / safe_arg
    signal_perp = j1_term ** 2
    
    # If argument is small, signal is 1.0
    signal_perp = jnp.where(valid_mask, signal_perp, 1.0)
    
    return signal_par * signal_perp


@jit
def c1_stick(bvals, bvecs, mu, lambda_par):
    """
    Computes signal for a Stick (zero-radius cylinder).
    
    Args:
        bvals: (N,) array of b-values in s/mm^2.
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (mm^2/s).
        
    Returns:
        (N,) array of signal attenuation (0.0 to 1.0).
    """
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    
    # Signal decay depends only on the parallel component
    # S = exp(-b * d_par * (g . mu)^2)
    signal = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    return signal
