import jax
import jax.numpy as jnp
import scipy.special as ssp
from jax.scipy import special as jsp
from jax import pure_callback, jit, vmap
from dmipy_jax.constants import GYRO_MAGNETIC_RATIO
import equinox as eqx
from jaxtyping import Array, Float
from typing import Any, Tuple


# Removed pure_callback wrappers. Using jax.scipy.special directly.


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
    
    j1_term = 2 * jsp.jn(1, safe_arg) / safe_arg
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


class RestrictedCylinder(eqx.Module):
    r"""
    The Stejskal-Tanner approximation of the cylinder model with finite radius [1]_.
    
    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in mm^2/s.
    diameter : float
        cylinder diameter in meters.

    References
    ----------
    .. [1] Soderman, Olle, and Bengt Jonsson. "Restricted diffusion in
            cylindrical geometry." Journal of Magnetic Resonance, Series A
            117.1 (1995): 94-97.
    """
    
    mu: Any = None
    lambda_par: Any = None
    diameter: Any = None

    parameter_names = ('mu', 'lambda_par', 'diameter')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'diameter': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'diameter': (1e-7, 20e-6)
    }

    def __init__(self, mu=None, lambda_par=None, diameter=None):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        diameter = kwargs.get('diameter', self.diameter)
        mu = kwargs.get('mu', self.mu)
        
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        
        if big_delta is None or small_delta is None:
             raise ValueError("RestrictedCylinder requires 'big_delta' and 'small_delta' in kwargs/acquisition.")

        # Convert spherical [theta, phi] to cartesian vector
        mu = jnp.asarray(mu)
        if mu.ndim > 0:
             theta = mu[0]
             phi = mu[1]
        else:
             theta = mu
             phi = 0.0 # Should not happen for RestrictedCylinder
             
        # jax.debug.print("mu shape: {}", mu.shape)
        
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        
        # Ensure mu_cart is (3,)
        mu_cart = jnp.array([st * cp, st * sp, ct])
        if mu_cart.ndim > 1:
            mu_cart = jnp.squeeze(mu_cart)
            
        jax.debug.print("RestrictedCylinder call: bvals {}, mu {}, mu_cart {}", bvals.shape, mu.shape, mu_cart.shape)

        return c2_cylinder(bvals, gradient_directions, mu_cart, lambda_par, diameter, big_delta, small_delta)


def jvp_v1(v, z):
    """
    Compute the first derivative of Bessel function of the first kind Jv(z) with respect to z.
    Formula: Jv'(z) = 0.5 * (J(v-1, z) - J(v+1, z))
    """
    return 0.5 * (jsp.bessel_jn(z, v=v - 1) - jsp.bessel_jn(z, v=v + 1))


@jit
def c3_cylinder_callaghan(bvals, bvecs, mu, lambda_par, diameter, diffusion_perp, tau, alpha):
    """
    Computes signal for a Cylinder with finite radius using Callaghan's approximation.
    
    Args:
        bvals: (N,) array of b-values in s/mm^2.
        bvecs: (N, 3) array of gradient directions.
        mu: (3,) array defining the fiber orientation.
        lambda_par: Scalar diffusivity along the fiber (mm^2/s).
        diameter: Cylinder diameter in meters.
        diffusion_perp: Perpendicular diffusivity (m^2/s) - typically same as intrinsic or smaller.
        tau: Diffusion time (s).
        alpha: (n_roots, n_functions) array of roots of J_n'(x)=0.
        
    Returns:
        (N,) array of signal attenuation.
    """
    # 1. Parallel Signal (Stick)
    # Project gradients onto fiber axis: (g . mu)
    dot_prod = jnp.dot(bvecs, mu)
    signal_par = jnp.exp(-bvals * lambda_par * (dot_prod ** 2))
    
    # 2. Perpendicular Signal (Callaghan)
    # Need q-values.
    # q = sqrt(b / tau) / (2pi) if we assume narrow pulse approximation for the q definition in Callaghan?
    # Legacy code uses q directly from acquisition.
    # q_mag = sqrt(b / tau) / (2pi) approx for Stejskal Tanner?
    # Actually, legacy uses: q = acquisition_scheme.qvalues
    # Here we typically start from bvals. 
    # Let's approximate q from bvals and tau assuming SGP or similar effective time.
    # q = 1/(2*pi) * sqrt(b/tau)
    # This matches the SGP relation b = (2pi q)^2 * tau.
    
    q_mag = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
    
    # Project gradients perpendicular to fiber axis
    sin_theta_sq = 1 - dot_prod**2
    sin_theta_sq = jnp.clip(sin_theta_sq, 0.0, 1.0) 
    q_perp = q_mag * jnp.sqrt(sin_theta_sq)
    
    radius = diameter / 2.0
    
    # Pre-calculate common terms
    # Pre-calculate common terms
    # Pre-calculate common terms
    q_argument = 2 * jnp.pi * q_perp * radius
    q_argument_2 = q_argument ** 2
    
    # Initialize response (summing updates)
    # We will accumulate signal attenuation.
    # Note: legacy code calculates the attenuation E_perp, then multiplies.
    
    # --- m = 0 case ---
    # alpha[k, 0] roots
    alpha_0 = alpha[:, 0] # (K,)
    
    # J0 term (actually eq uses J1 for m=0 term update?)
    # Legacy: J = special.j1(q_argument) ** 2
    # update = 4 * exp(...) * q_arg^2 / (q_arg^2 - alpha^2)^2 * J
    
    J_m0 = jsp.bessel_jn(q_argument, v=1) ** 2 # (N,)
    
    # Vectorized compute over K roots for m=0
    # exp_factor: (K,) scalar (per root)
    # But tau is (N,) or scalar? usually scalar per shell, but can be array.
    # Let's assume tau is scalar for now or broadcastable.
    
    # We need to broadcast (N, K).
    # q_argument: (N,)
    # alpha_0: (K,)
    
    # Reshape for broadcasting
    q_arg_2_expanded = q_argument_2[:, None] # (N, 1)
    alpha_0_expanded = alpha_0[None, :]      # (1, K)
    alpha_0_sq = alpha_0_expanded ** 2
    
    # exp_term: (1, K) usually, if tau is scalar. If tau is (N,), then (N, K).
    # diffusion_perp is scalar. radius is scalar.
    exp_term_0 = jnp.exp(-alpha_0_sq * diffusion_perp * tau / (radius ** 2))
    
    denom_0 = (q_arg_2_expanded - alpha_0_sq) ** 2
    
    # Prevent division by zero if q_arg ~ alpha (resonance)
    denom_0 = jnp.maximum(denom_0, 1e-12)
    
    term_0 = (4 * exp_term_0 * q_arg_2_expanded / denom_0)
    
    # Sum over k
    sum_0 = jnp.sum(term_0, axis=1) # (N,)
    
    res = sum_0 * J_m0
    
    # --- m > 0 cases ---
    # We loop over m efficiently.
    # Since m ranges 1..50, we can Python-loop and accumulate.
    # It adds nodes to the graph but 50 is manageable.
    
    n_functions = alpha.shape[1]
    
    def loop_body(m, accumulated_res):
        # alpha for this m
        alpha_m = alpha[:, m] # (K,)
        alpha_m_sq = alpha_m[None, :] ** 2 # (1, K)
        
        # J term: J'm(q_arg)
        # Legacy: J = special.jvp(m, q_argument, 1)
        J_val = jvp_v1(m, q_argument) # (N,)
        
        q_arg_J = (q_argument * J_val) ** 2 # (N,)
        q_arg_J_expanded = q_arg_J[:, None] # (N, 1)
        
        # Denom
        denom_m = (q_arg_2_expanded - alpha_m_sq) ** 2
        denom_m = jnp.maximum(denom_m, 1e-12)
        
        # exp term
        exp_term_m = jnp.exp(-alpha_m_sq * diffusion_perp * tau / (radius ** 2))
        
        # Update term
        # 8 * exp(...) * alpha^2 / (alpha^2 - m^2) * q_arg_J / denom
        
        numerator_factor = alpha_m_sq / (alpha_m_sq - m**2) # (1, K)
        
        term_m = (8 * exp_term_m * numerator_factor * q_arg_J_expanded / denom_m)
        
        sum_m = jnp.sum(term_m, axis=1) # (N,)
        
        return accumulated_res + sum_m
    
    # Iterate m from 1 to n_functions - 1
    # We can use jax.lax.fori_loop if we want to condense the graph, 
    # but that requires 'm' to be dynamic in 'bessel_jn' or handled carefully.
    # bessel_jn supports float v, so dynamic m (as generic int/float) in scan/fori should work.
    # However, alpha slice `alpha[:, m]` needs dynamic index.
    
    # Using python loop for now to be safe + explicit
    for m in range(1, int(n_functions)):
        res = loop_body(m, res)
        
    # Handle q_perp = 0 case (no attenuation perpendicular)
    res = jnp.where(q_perp > 1e-9, res, 1.0)
    
    return signal_par * res


class CallaghanRestrictedCylinder(eqx.Module):
    r"""
    The Callaghan model [1]_ - a cylinder with finite radius - typically
    used for intra-axonal diffusion. The perpendicular diffusion is modelled
    after Callaghan's solution for the disk. Is dependent on both q-value
    and diffusion time.

    Parameters
    ----------
    mu : array, shape(2)
        angles [theta, phi] representing main orientation on the sphere.
    lambda_par : float
        parallel diffusivity in mm^2/s.
    diameter : float
        cylinder (axon) diameter in meters.
    diffusion_perpendicular : float
        intra-cylindrical perpendicular diffusivity (m^2/s).
    number_of_roots : integer
        number of roots to use for the Callaghan cylinder model.
    number_of_functions : integer
        number of functions to use for the Callaghan cylinder model.

    References
    ----------
    .. [1] Callaghan, Paul T. "Pulsed-gradient spin-echo NMR for planar,
            cylindrical, and spherical pores under conditions of wall
            relaxation." Journal of magnetic resonance, Series A 113.1 (1995):
            53-59.
    """
    
    mu: Any = None
    lambda_par: Any = None
    diameter: Any = None
    diffusion_perpendicular: Any = 1.7e-9
    number_of_roots: int = eqx.field(static=True, default=20)
    number_of_functions: int = eqx.field(static=True, default=50)
    # alpha should be a field to be part of the pytree
    alpha: Array = eqx.field(init=False)

    parameter_names = ('mu', 'lambda_par', 'diameter', 'diffusion_perpendicular')
    parameter_cardinality = {'mu': 2, 'lambda_par': 1, 'diameter': 1, 'diffusion_perpendicular': 1}
    parameter_ranges = {
        'mu': ([0, jnp.pi], [-jnp.pi, jnp.pi]),
        'lambda_par': (0.1e-9, 3e-9),
        'diameter': (1e-7, 20e-6),
        'diffusion_perpendicular': (0.1e-9, 3e-9)
    }

    def __init__(self, mu=None, lambda_par=None, diameter=None, 
                 diffusion_perpendicular=1.7e-9, number_of_roots=20, number_of_functions=50):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diameter = diameter
        self.diffusion_perpendicular = diffusion_perpendicular
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions
        
        # Pre-calculate alpha (roots) using scipy.special (on CPU/host)
        self.alpha = self._precompute_roots(number_of_roots, number_of_functions)

    def _precompute_roots(self, n_roots, n_functions):
        import numpy as np
        # Use numpy for this ensuring it runs at instantiation
        alpha = np.empty((n_roots, n_functions))
        alpha[0, 0] = 0
        if n_roots > 1:
            alpha[1:, 0] = ssp.jnp_zeros(0, n_roots - 1)
        for m in range(1, n_functions):
            alpha[:, m] = ssp.jnp_zeros(m, n_roots)
        return jnp.array(alpha)

    def __call__(self, bvals, gradient_directions, **kwargs):
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        diameter = kwargs.get('diameter', self.diameter)
        diffusion_perp = kwargs.get('diffusion_perpendicular', self.diffusion_perpendicular)
        mu = kwargs.get('mu', self.mu)
        
        # Need tau (diffusion time)
        # Prefer 'tau' from kwargs, else derive from big_delta/small_delta
        if 'tau' in kwargs:
            tau = kwargs['tau']
        elif 'big_delta' in kwargs and 'small_delta' in kwargs:
             tau = kwargs['big_delta'] - kwargs['small_delta'] / 3.0
        else:
             raise ValueError("CallaghanRestrictedCylinder requires 'tau' or 'big_delta'/'small_delta' in kwargs.")

        # Convert spherical [theta, phi] to cartesian vector
        theta = mu[0]
        phi = mu[1]
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        sp = jnp.sin(phi)
        cp = jnp.cos(phi)
        mu_cart = jnp.array([st * cp, st * sp, ct])

        return c3_cylinder_callaghan(
            bvals, gradient_directions, mu_cart, lambda_par, diameter, diffusion_perp, tau, self.alpha
        )

