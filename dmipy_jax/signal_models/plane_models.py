
import jax.numpy as jnp
from jax import jit, lax

@jit
def g2_plane_stejskal_tanner(q, diameter):
    """
    Stejskal-Tanner approximation of diffusion between two infinitely large parallel planes.
    """
    q_argument = 2 * jnp.pi * q * diameter
    
    # Handle singularity at q=0
    # cos(x) ~ 1 - x^2/2. (1 - cos) ~ x^2/2.
    # 2 * (x^2/2) / x^2 = 1.
    
    # Safe division
    is_nonzero = jnp.abs(q_argument) > 1e-6
    safe_arg = jnp.where(is_nonzero, q_argument, 1.0)
    
    res = 2 * (1 - jnp.cos(safe_arg)) / (safe_arg ** 2)
    
    return jnp.where(is_nonzero, res, 1.0)


class PlaneStejskalTanner:
    r"""
    Stejskal-Tanner approximation of diffusion between two infinitely large parallel planes.
    """
    
    parameter_names = ['diameter']
    parameter_cardinality = {'diameter': 1}
    parameter_ranges = {'diameter': (1e-6, 20e-6)}

    def __init__(self, diameter=None):
        self.diameter = diameter

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        
        if 'qvalues' in kwargs:
             q = kwargs['qvalues']
        elif bvals is not None:
             # Check for timing
             if 'big_delta' in kwargs and 'small_delta' in kwargs:
                 tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
                 q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
             elif 'tau' in kwargs:
                 q = jnp.sqrt(bvals / (kwargs['tau'] + 1e-12)) / (2 * jnp.pi)
             else:
                  # If bvals are zero, q is zero.
                  if jnp.all(bvals == 0):
                       q = jnp.zeros_like(bvals)
                  else:
                       raise ValueError("PlaneStejskalTanner requires 'qvalues' or timing.")
        else:
             raise ValueError("PlaneStejskalTanner requires 'qvalues' or bvals+timing.")
             
        return g2_plane_stejskal_tanner(q, diameter)


@jit
def g3_plane_callaghan(q, tau, diameter, diffusion_constant, xi, zeta):
    """
    Callaghan approximation for planes.
    xi: roots n*pi
    zeta: roots (n+1/2)*pi
    """
    radius = diameter / 2.0
    q_argument = 2 * jnp.pi * q * radius
    q_argument_2 = q_argument ** 2
    
    res = jnp.zeros_like(q)
    
    # Loop over xi (n=0..N-1)
    # Roots xi = n*pi.
    
    def xi_loop(carry, n):
        # We assume xi array is passed correctly
        xi_n = xi[n]
        xi_n2 = xi_n**2
        
        # div = sin(2*xi)/(2*xi). If xi=0 -> 1.
        is_zero = jnp.abs(xi_n) < 1e-9
        safe_xi = jnp.where(is_zero, 1.0, xi_n)
        div = jnp.sin(2 * safe_xi) / (2 * safe_xi)
        div = jnp.where(is_zero, 1.0, div)
        
        # update
        # term 1: exp
        E_time = jnp.exp(-xi_n2 * diffusion_constant * tau / radius ** 2)
        
        # term 2: geometric
        # (q_arg * sin(q_arg) * cos(xi) - xi * cos(q_arg) * sin(xi))^2 / (q_arg^2 - xi^2)^2
        # note: sin(xi)=0 for xi=n*pi. cos(xi) = +/- 1.
        # simpler: (q_arg * sin(q_arg) * (+/-1) - 0)^2 = (q_arg * sin(q_arg))^2 ?
        # Wait, if xi=pi, sin(xi)=0. cos(xi)=-1.
        # Indeed, sin(n*pi)=0.
        # So the second part of numerator is 0.
        # result: (q_arg * sin(q_arg) * cos(xi))^2 / (q_arg^2 - xi^2)^2
        #       = q_arg^2 * sin^2(q_arg) * 1 / ...
        # Is this simplification safe? "xi" in legacy is `np.arange * np.pi`. Yes.
        # But we implement the full formula to be safe against float drift?
        
        bracket = (q_argument * jnp.sin(q_argument) * jnp.cos(xi_n) 
                   - xi_n * jnp.cos(q_argument) * jnp.sin(xi_n))
        
        denom_geom = (q_argument_2 - xi_n2) ** 2
        
        # Handle resonance denom=0? (q_arg = xi)
        denom_geom = jnp.maximum(denom_geom, 1e-12)
        
        term = 2 * E_time / (1 + div) * (bracket ** 2) / denom_geom
        
        return carry + term, None

    sum_xi, _ = lax.scan(xi_loop, jnp.zeros_like(res), jnp.arange(xi.shape[0]))
    res += sum_xi
    
    # Loop over zeta
    def zeta_loop(carry, m):
        zeta_m = zeta[m]
        zeta_m2 = zeta_m**2
        
        # div = sin(2zeta)/(2zeta). zeta always > 0 (pi/2 ...)
        div = jnp.sin(2 * zeta_m) / (2 * zeta_m)
        
        E_time = jnp.exp(-zeta_m2 * diffusion_constant * tau / radius ** 2)
        
        # bracket: q*cos(q)*sin(zeta) - zeta*sin(q)*cos(zeta)
        # sin(zeta) = sin((n+0.5)pi) = +/- 1.
        # cos(zeta) = 0.
        # So second term vanishes? 
        # Yes, cos((n+0.5)pi) = 0.
        
        bracket = (q_argument * jnp.cos(q_argument) * jnp.sin(zeta_m) 
                   - zeta_m * jnp.sin(q_argument) * jnp.cos(zeta_m))
                   
        denom_geom = (q_argument_2 - zeta_m2) ** 2
        denom_geom = jnp.maximum(denom_geom, 1e-12)
        
        term = 2 * E_time / (1 - div) * (bracket ** 2) / denom_geom
        
        return carry + term, None

    sum_zeta, _ = lax.scan(zeta_loop, jnp.zeros_like(res), jnp.arange(zeta.shape[0]))
    res += sum_zeta
    
    return res


class PlaneCallaghan:
    r"""
    The Callaghan model [1]_ of diffusion between two parallel infinite plates.
    """
    
    parameter_names = ['diameter', 'diffusion_constant']
    parameter_cardinality = {'diameter': 1, 'diffusion_constant': 1}
    parameter_ranges = {
        'diameter': (1e-6, 20e-6),
        'diffusion_constant': (0.1e-9, 3e-9)
    }

    def __init__(self, diameter=None, diffusion_constant=None, number_of_roots=40):
        self.diameter = diameter
        self.diffusion_constant = diffusion_constant
        self.number_of_roots = number_of_roots
        
        self.xi = jnp.arange(number_of_roots) * jnp.pi
        self.zeta = jnp.arange(number_of_roots) * jnp.pi + jnp.pi / 2.0

    def __call__(self, bvals, gradient_directions, **kwargs):
        diameter = kwargs.get('diameter', self.diameter)
        diff_const = kwargs.get('diffusion_constant', self.diffusion_constant)
        
        if 'qvalues' in kwargs:
             q = kwargs['qvalues']
        elif bvals is not None and 'big_delta' in kwargs and 'small_delta' in kwargs:
             tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
             q = jnp.sqrt(bvals / (tau + 1e-12)) / (2 * jnp.pi)
        elif bvals is not None and 'tau' in kwargs:
             q = jnp.sqrt(bvals / (kwargs['tau'] + 1e-12)) / (2 * jnp.pi)
        else:
             raise ValueError("PlaneCallaghan requires 'qvalues' or bvals+timing.")
             
        if 'tau' in kwargs:
            tau = kwargs['tau']
        elif 'big_delta' in kwargs and 'small_delta' in kwargs:
            tau = kwargs['big_delta'] - kwargs['small_delta']/3.0
        else:
             raise ValueError("PlaneCallaghan requires 'tau'.")
             
        return g3_plane_callaghan(q, tau, diameter, diff_const, self.xi, self.zeta)
