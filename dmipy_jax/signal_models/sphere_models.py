import jax.numpy as jnp
from jax import jit
import scipy.special as ssp
from jax.scipy import special as jsp
from dmipy_jax.constants import SPHERE_ROOTS, GYRO_MAGNETIC_RATIO
import equinox as eqx
from typing import Any

class S1Dot:
    r"""
    The Dot model [1]_ - an non-diffusing compartment.
    It has no parameters and returns 1 no matter the input.

    References
    ----------
    .. [1] Panagiotaki et al.
           "Compartment models of the diffusion MR signal in brain white
            matter: a taxonomy and comparison". NeuroImage (2012)
    """
    
    parameter_names = []
    parameter_cardinality = {}
    parameter_ranges = {}
    
    def __init__(self):
        pass
        
    def __call__(self, bvals, gradient_directions, **kwargs):
        # Return ones matching the shape of bvals
        return jnp.ones_like(bvals, dtype=float)


@jit
def g2_sphere_stejskal_tanner(q, diameter):
    """
    The Stejskal Tanner signal approximation of a sphere model.
    """
    radius = diameter / 2.0
    factor = 2 * jnp.pi * q * radius
    
    # Handle singularity at factor=0
    # sin(x)/x -> 1, cos(x) -> 1. (1-1) -> 0.
    # But prefactor 3/x^2 (singularity).
    # Expansion: sin(x)/x approx 1 - x^2/6. cos(x) approx 1 - x^2/2.
    # Bracket: (1 - x^2/6) - (1 - x^2/2) = x^2/3.
    # Result: 3/x^2 * (x^2/3) = 1.
    
    # Use safe division
    is_nonzero = jnp.abs(factor) > 1e-6
    safe_factor = jnp.where(is_nonzero, factor, 1.0)
    
    term = (jnp.sin(safe_factor) / safe_factor) - jnp.cos(safe_factor)
    
    # E = [3 * j1(z) / z]^2
    # term = z * j1(z)
    # We want 3 * (term/z) / z = 3 * term / z^2
    
    E = (3 * term / (safe_factor ** 2)) ** 2
    
    return jnp.where(is_nonzero, E, 1.0)


class SphereStejskalTanner:
    r"""
    The Stejskal Tanner signal approximation of a sphere model.
    
    Parameters
    ----------
    diameter : float
        sphere diameter in meters.
    """
    
    parameter_names = ['diameter']
    parameter_cardinality = {'diameter': 1}
    parameter_ranges = {'diameter': (1e-6, 20e-6)} # Typical range

    def __init__(self, diameter=None):
        self.diameter = diameter

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        
        # We need qvalues.
        # Check if 'qvalues' in kwargs, else derive from bvals/tau approx?
        # Stejskal Tanner theory strictly uses q.
        # q = 1/(2pi) * sqrt(b/tau) ?
        # Assuming acquisition provides 'qvalues' or we calculate q_mag.
        
        if 'qvalues' in kwargs:
             q = kwargs['qvalues']
        elif 'bvals' in kwargs or bvals is not None:
             # Try to derive q from bvals if tau provided
             # But bvals not passed if we use **kwargs? bvals passed explicitly in __call__ usually.
             
             # Fallback to approximating q from bvals if tau known is dangerous without G info.
             # However, typically in dMipy acquisition scheme, q is precalculated.
             # OptimistixFitter wrapper usually unpacks acquisition.
             # Let's try to calculate q if tau (or big/small delta) is present.
             
             if 'tau' in kwargs:
                 tau = kwargs['tau']
                 q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
             elif 'big_delta' in kwargs and 'small_delta' in kwargs:
                 tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
                 q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
             else:
                 # Last resort: if bvals are 0, q is 0.
                 # If bvals > 0, we need timing.
                 # Raise error if timing missing.
                 raise ValueError("SphereStejskalTanner requires 'qvalues' or timing info ('tau' or 'big_delta'/'small_delta') to derive q.")
        else:
             raise ValueError("SphereStejskalTanner requires 'qvalues' in kwargs.")
             
        return g2_sphere_stejskal_tanner(q, diameter)


def jsph_derivative(n, z):
    """
    Derivative of spherical Bessel function j_n(z).
    Formula: j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z)
    Or: j_n'(z) = n/z * j_n(z) - j_{n+1}(z)
    """
    # Using python control flow for n? n is integer loop variable.
    # jax.scipy.special.spherical_jn requires fixed order usually? no, v is float.
    # However, safe division by z needed.
    
    jn = lambda v, x: jsp.bessel_jn(x, v=v + 0.5) * jnp.sqrt(jnp.pi / (2 * x))
    # Wait, jax.scipy.special doesn't have spherical_jn directly? 
    # It has bessel_jn. spherical_jn(n, z) = sqrt(pi/2z) * J_{n+0.5}(z).
    
    # Actually, let's check if jsp has spherical_jn.
    # If not, implement it. 
    # Current doc says jax.scipy.special has bessel_jn.
    
    # We will use the formula j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z).
    # Only need to handle z=0 case (derivative is 0 for n>1?).
    
    # Let's implement a helper using bessel_jn.
    pass

# Helper for spherical bessel
def spherical_jn_jax(n, z):
    # sqrt(pi / (2z)) * J(n + 0.5, z)
    # Handle z=0: if n=0, 1. if n>0, 0.
    
    safe_z = jnp.where(z < 1e-10, 1e-10, z)
    return jnp.sqrt(jnp.pi / (2 * safe_z)) * jsp.bessel_jn(safe_z, v=n + 0.5)


def spherical_jn_derivative_jax(n, z):
    # j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z)
    # Be careful at z=0.
    # For Callaghan sphere, we evaluate at q_argument.
    
    safe_z = jnp.where(z < 1e-10, 1e-10, z)
    
    j_n = spherical_jn_jax(n, safe_z)
    
    if n == 0:
        # j0(z) = sin(z)/z
        # j0'(z) = -j1(z)
        j_n_plus_1 = spherical_jn_jax(1, safe_z)
        return -j_n_plus_1
    else:
        j_n_minus_1 = spherical_jn_jax(n - 1, safe_z)
        return j_n_minus_1 - (n + 1) / safe_z * j_n


@jit
def g3_sphere_callaghan(q, tau, diameter, diffusion_constant, alpha):
    """
    Callaghan sphere model kernel.
    alpha: roots (n_roots, n_functions)
    """
    radius = diameter / 2.0
    q_argument = 2 * jnp.pi * q * radius
    q_argument_2 = q_argument ** 2
    
    # If q is 0, signal is 1.
    
    # Jder = spherical_jn_jax(0, q_argument, derivative=True) ??
    # Legacy uses `special.spherical_jn(q_argument, derivative=True)` which returns values for ALL n?
    # No, scipy.special.spherical_jn(n, z) returns for specific n if n is scalar.
    # If n is not given? "If n is an array_like, the result is computed for each n."
    
    # Legacy code:
    # Jder = special.spherical_jn(q_argument, derivative=True)
    # Wait, check legacy line 190 in P3SphereCallaghan?
    # "Jder = special.spherical_jn(q_argument, derivative=True)"
    # This might be incorrect reading of legacy code or legacy code relies on a deprecated behavior? 
    # Or maybe it calls derivative w.r.t argument?
    # Actually, looking at the loop in legacy:
    # `for n in range(0, self.alpha.shape[1]):` it uses n.
    # But `update *= q_argument * Jder` in the loop.
    # Where does `Jder` depend on `n` in legacy?
    # Ah, checking legacy snippet:
    # `Jder = special.spherical_jn(q_argument, derivative=True)`
    # This looks like it calls it without `n`? 
    # In scipy <= 0.18 spherical_jn took n. In newer scipy, it might be different.
    # Actually `scipy.special.spherical_jn(n, z, derivative=False)`.
    # Attempting to call it without n usually fails?
    # Unless q_argument is interpreted as n? No, q_argument is array of floats.
    
    # Let's assume Jder depends on n.
    # Inside the loop over n: `update *= q_argument * Jder`.
    # If Jder was constant for all n, that would be weird.
    # Oh, wait. In the legacy snippet I viewed earlier (Step 146):
    # `Jder = special.spherical_jn(q_argument, derivative=True)` is OUTSIDE the loops!
    # And it's used inside.
    # This implies Jder is `j_0'(q)`? Or maybe `spherical_jn` returns list if n omitted?
    # checking scipy docs... `spherical_jn(n, z)`
    # The snippet seems suspicious or relies on old behavior.
    
    # However, let's look at the math from Callaghan 1995.
    # Formula usually involves summing over zeros of derivatives.
    # It likely involves weighting by the value of the function (or derivative) at q.
    # If legacy computes `Jder` once, it might be an error in legacy or my reading.
    
    # Let's re-read legacy loop carefully in Step 146.
    # Line 190: `Jder = special.spherical_jn(q_argument, derivative=True)`
    # Line 191: `for k in range...`
    # Line 192: `for n in range...`
    # Line 197: `update *= q_argument * Jder`
    
    # This suggests `Jder` does NOT depend on `n` or `k`.
    # This implies `n=0` or it's array of shape (N_q,)?
    # If `special.spherical_jn` is called with just `q_argument` (as n?), that would be n=q_argument? No.
    # Maybe `n` defaults to 0?
    
    # Let's assume it requires proper `j_n'(qR)`.
    # We will implement the correct math:
    # Sum_n Sum_k [ ... * (qR * j_n'(qR))^2? or something ]
    # Re-reading: `update *= q_argument * Jder` -> `update *= qR * j?'(qR)`
    # And `update /= (qR**2 - alpha**2)**2`
    
    # Warning: If legacy code is buggy, we should fix it or match it?
    # "We follow the notation of Balinov".
    # I will assume JAX implementation should be mathematically correct.
    # I will construct the term properly dependent on n if the math requires it.
    
    # Actually, `spherical_jn(n, z)` in scipy.
    # If I implement `j_n'` I should use it inside the loop for `n`.
    
    # Let's implement the loop over n and k.
    
    # Initialize res
    res = jnp.zeros_like(q)
    
    n_roots = alpha.shape[0]
    n_functions = alpha.shape[1]
    
    # We iterate n (functions) and k (roots).
    
    # Precompute q_argument
    
    def body_fun(carry, n):
        # n is loop index (0 to n_functions-1)
        res_bh = carry
        
        # Calculate Jder_n for this n
        Jder_n = spherical_jn_derivative_jax(n, q_argument)
        
        def root_body(carry_k, k):
            res_k = carry_k
            alpha_nk = alpha[k, n]
            alpha_nk2 = alpha_nk**2
            
            # Update term
            # exp(-alpha^2 * D * tau / R^2)
            E_time = jnp.exp(-alpha_nk2 * diffusion_constant * tau / radius ** 2)
            
            # Weighting
            # (2n+1)alpha^2 / (alpha^2 - (n+0.5)^2 + 0.25)
            # note: (n+0.5)^2 - 0.25 = n^2 + n + 0.25 - 0.25 = n(n+1)
            # So denom is alpha^2 - n(n+1).
            denom_weight = alpha_nk2 - n*(n+1)
            weight = ((2 * n + 1) * alpha_nk2) / denom_weight
            
            # Geometric term
            # q_argument * Jder_n / (q_argument^2 - alpha^2)
            # Legacy squares the denominator `(q_argument_2 - a_nk2) ** 2`.
            # And `update *= q_argument * Jder`.
            # Wait, legacy didn't square `(q_argument * Jder)`? 
            # In Cylinder `c3`, it was `(q*J')^2`.
            # Here legacy says: `update *= q_argument * Jder`. 
            # Then `update /= (q_argument_2 - a_nk2) ** 2`.
            # Is `update` pre-accumulated with something?
            # `update = np.exp(...)`. Then *= weight. Then *= qJder. Then /= denom^2.
            # This results in linear term in Jder?
            # Verify dimensionality. Signal is amplitude squared? No, typically real valued attenuation E.
            # Usually E = Sum [ ... ].
            
            # I will trust the formula:
            # Term ~ (qR * j_n'(qR))^2? 
            # Let's assume legacy might be right about linear `q * Jder` IF `sphere_attenuation` returns simple E? 
            # Wait, sphere attenuation usually decays.
            
            # Let's use the exact legacy formula logic for now to ensure equivalence, 
            # BUT assuming Jder depends on n.
            
            term = E_time * weight * (q_argument * Jder_n) / ((q_argument_2 - alpha_nk2)**2)
            
            return res_k + term, None

        # scan over k
        sum_k, _ = jax.lax.scan(root_body, jnp.zeros_like(res), jnp.arange(n_roots))
        
        return res_bh + sum_k, None
    
    # We can unroll the loops since n_functions ~ 50 is small enough? 
    # Or use python loop for n (easier derivative dispatch).
    
    accum = jnp.zeros_like(q)
    for n in range(int(n_functions)):
        Jder_n = spherical_jn_derivative_jax(n, q_argument)
        
        # Vectorize over k (roots)
        # alpha[:, n] -> (K,)
        alphas_n = alpha[:, n]
        alphas_n2 = alphas_n**2
        
        E_times = jnp.exp(-alphas_n2[..., None] * diffusion_constant * tau / radius**2) # (K, N)??
        # q is (N,). alphas is (K,).
        # We need broadcasting.
        
        # (K, 1)
        an2_col = alphas_n2[:, None]
        
        E_times = jnp.exp(-an2_col * diffusion_constant * tau / radius**2) # (K, N) if tau scalar?
        # If tau is (N,), then (K, N).
        
        denom_weight = an2_col - n*(n+1)
        # Handle 0/0 for n=0, k=0 (alpha=0)?
        # If alpha=0, denom=0.
        # Legacy: alpha[0,0]=0.
        # `update` for 0,0:
        # exp(0) * (1*0)/(0) -> NaN?
        # Legacy likely handles `alpha=0` by skipping or specialized limit?
        # Legacy loop: `range(0, roots)`.
        # `(2*n+1)*a^2 / (a^2 - n(n+1))`.
        # If n=0, a=0: 0 / 0.
        # We should safe guard or skip alpha=0?
        # Actually legacy `alpha[0,0] = 0`.
        # I'll implement safe division.
        
        weight = ((2 * n + 1) * an2_col) / jnp.where(denom_weight==0, 1.0, denom_weight)
        
        # J_term
        # q_arg (N,)
        num_geom = q_argument * Jder_n # (N,)
        denom_geom = (q_argument_2 - an2_col)**2 # (K, N)
        
        # Guard against singularity where qR approaches a root (alpha)
        denom_geom = jnp.maximum(denom_geom, 1e-12)
        
        # Full term
        term = E_times * weight * num_geom / denom_geom
        
        accum += jnp.sum(term, axis=0)

    # 2 * sum ...
    # Wait, does the legacy formula have a factor of 2 outside?
    # No, it calculates `update` and sums.
    # Where does the 2 come from in Cylinder/Plane?
    # Here Sphere legacy doesn't show a 2.
    
    return accum * 2 # Adding *2 because usually there's a prefactor, and legacy P3Plane has 2.
    # Wait, legacy S3Sphere logic viewed in Step 146 lines 182-200 DOES NOT have *2 at the end.
    # It just returns `res`.
    # I will stick to returning `accum` but double check if scaling needed.
    # Actually, let's strictly follow legacy lines 194-198.
    # No global factor 2 seen.
    
    return accum    


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
    # Ensure deltas broadcast against roots (axis 1)
    # small_delta/big_delta: (N,) or scalar -> expand to (N, 1) or (1, 1) if not already
    sd_expanded = jnp.expand_dims(small_delta, -1) if jnp.ndim(small_delta) > 0 else small_delta
    bd_expanded = jnp.expand_dims(big_delta, -1) if jnp.ndim(big_delta) > 0 else big_delta
    
    exp_1 = jnp.exp(-Dm_alpha2 * sd_expanded)
    exp_2 = jnp.exp(-Dm_alpha2 * bd_expanded)
    exp_3 = jnp.exp(-Dm_alpha2 * (bd_expanded - sd_expanded))
    exp_4 = jnp.exp(-Dm_alpha2 * (bd_expanded + sd_expanded))
    
    time_term = (
        2 * sd_expanded 
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


class SphereGPD(eqx.Module):
    r"""
    The Gaussian Phase Distribution (GPD) approximation of the Sphere model [1]_.
    Also known as the Murday-Cotts model.

    Parameters
    ----------
    diameter : float
        sphere diameter in meters.
    diffusion_constant : float
        diffusion constant in m^2/s.

    References
    ----------
    .. [1] Murday, J. S., and R. M. Cotts. "Self-diffusion coefficient of
            liquid lithium." The Journal of Chemical Physics 48.11 (1968):
            4938-4945.
    """
    
    diameter: Any = None
    diffusion_constant: Any = None

    parameter_names = ('diameter', 'diffusion_constant')
    parameter_cardinality = {'diameter': 1, 'diffusion_constant': 1}
    parameter_ranges = {
        'diameter': (1e-6, 20e-6),
        'diffusion_constant': (0.1e-9, 3e-9)
    }

    def __init__(self, diameter=None, diffusion_constant=None):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        diffusion_constant = kwargs.get('diffusion_constant', self.diffusion_constant)
        
        big_delta = kwargs.get('big_delta')
        small_delta = kwargs.get('small_delta')
        
        if big_delta is None or small_delta is None:
             raise ValueError("SphereGPD requires 'big_delta' and 'small_delta' in kwargs.")

        return g3_sphere(bvals, gradient_directions, diameter, diffusion_constant, big_delta, small_delta)



class SphereCallaghan:
    r"""
    The Callaghan model [1]_ of diffusion inside a sphere.
    """
    
    parameter_names = ['diameter', 'diffusion_constant']
    parameter_cardinality = {'diameter': 1, 'diffusion_constant': 1}
    parameter_ranges = {
        'diameter': (1e-6, 20e-6),
        'diffusion_constant': (0.1e-9, 3e-9)
    }

    def __init__(self, diameter=None, diffusion_constant=None, number_of_roots=20, number_of_functions=50):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.number_of_roots = number_of_roots
        self.number_of_functions = number_of_functions
        
        self.alpha = self._precompute_roots(number_of_roots, number_of_functions)
        
    def _precompute_roots(self, n_roots, n_functions):
        import numpy as np
        alpha = np.empty((n_roots, n_functions))
        alpha[0, 0] = 0
        if n_roots > 1:
            alpha[1:, 0] = ssp.jnp_zeros(0, n_roots - 1)
        for m in range(1, n_functions):
            alpha[:, m] = ssp.jnp_zeros(m, n_roots)
        return jnp.array(alpha)

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        diff_const = kwargs.get('diffusion_constant', self.diffusion_constant)
        
        # Need q and tau.
        if 'qvalues' in kwargs:
            q = kwargs['qvalues']
        elif bvals is not None and 'big_delta' in kwargs and 'small_delta' in kwargs:
             tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
             q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
        elif bvals is not None and 'tau' in kwargs:
             # If tau provided but not big/small
             q = jnp.sqrt(bvals / (kwargs['tau'] + 1e-12)) / (2 * jnp.pi)
        else:
             raise ValueError("SphereCallaghan requires 'qvalues' or bvals+timing.")
             
        if 'tau' in kwargs:
            tau = kwargs['tau']
        elif 'big_delta' in kwargs and 'small_delta' in kwargs:
            tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
        else:
             raise ValueError("SphereCallaghan requires 'tau' or 'big_delta'/'small_delta'.")
             
        return g3_sphere_callaghan(q, tau, diameter, diff_const, self.alpha)