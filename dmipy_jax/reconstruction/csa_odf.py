import jax
import jax.numpy as jnp
from dmipy_jax.utils.spherical_harmonics import fit_spherical_harmonics, sh_basis_real_analytic

def compute_csa_odf_weights(lmax):
    """
    Computes the SH spectral weights for CSA-ODF.
    Weight_l = (C_l * L_l) / (16 * pi^2)
    
    where:
    L_l = -l(l+1) (Laplace-Beltrami eigenvalue)
    C_l = 2*pi * P_l(0) (Funk-Radon transform factor)
    """
    weights = []
    
    # Precompute factorials/products?
    # l is even. 0, 2, 4...
    
    for l in range(0, lmax + 1, 2):
        # L_l
        L_val = -l * (l + 1)
        
        # C_l
        # P_l(0) = (-1)^(l/2) * (l-1)!! / l!!
        # double factorial ratio
        
        if l == 0:
            P_l0 = 1.0
        else:
            # prod(1:2:(k-1)) / prod(2:2:k)
            numer = 1.0
            denom = 1.0
            for i in range(1, l, 2): numer *= i
            for i in range(2, l + 1, 2): denom *= i
            P_l0 = ((-1)**(l // 2)) * numer / denom
            
        C_val = 2 * jnp.pi * P_l0
        
        # CSA Weight
        w = (C_val * L_val) / (16 * jnp.pi**2)
        
        # Repeat for all m in -l..l (2l+1 times)
        n_m = 2 * l + 1
        weights.append(jnp.full((n_m,), w))
        
    return jnp.concatenate(weights)

def fit_csa_odf(signal, acquisition, lmax=4, delta=1e-4):
    """
    Reconstructs CSA-ODF coefficients.
    
    1. Transform Signal: y = log(-log(S/S0))
       (Requires normalized signal S/S0).
    2. Fit SH to y.
    3. Apply CSA weights (Laplacian + Funk-Radon).
    
    Args:
        signal: S (or S/S0). If S, ensure normalized? 
                The Aganj code assumes S/S0 input to log(-log).
        acquisition: JaxAcquisition scheme.
    """
    # 1. Transform Signal
    # Handle edges: S/S0 should be < 1.
    # signal is assumed to be S/S0 (normalized).
    
    # Clip signal to avoid NaN in log(-log)
    # log(-log(1)) = log(0) = -inf
    # log(-log(0)) = log(inf) = inf
    # range approx (0.001, 0.999)
    # Aganj uses reg.m to regularize: S = max(S, delta)?
    # reconCSAODF.m line 136: rE = reg(S/S0, delta)
    # reg.m probably does standard flooring.
    
    s_clipped = jnp.clip(signal, delta, 1.0 - delta)
    y = jnp.log(-jnp.log(s_clipped))
    
    # 2. Fit SH to y
    coeffs_y = fit_spherical_harmonics(y, acquisition, lmax)
    
    # 3. Apply Weights
    weights = compute_csa_odf_weights(lmax)
    
    # Broadcast weights: (N_coeffs,) -> (..., N_coeffs)
    odf_coeffs = coeffs_y * weights
    
    # GFA/visualisation typically requires ODF(u) > 0.
    # But coefficients themselves are just SH coeffs.
    
    return odf_coeffs
