import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import equinox as eqx
from jaxtyping import Float, Array
from typing import Optional, Any

class KargerExchangeModel(eqx.Module):
    """
    Kärger Model for 2-compartment exchange.
    
    Models diffusion between two compartments (Intra/Extra) with exchange.
    Uses Matrix Exponential to solve the coupled differential equations.
    
    Parameters:
        D_intra (float): Diffusivity of intra-cellular compartment (m^2/s).
        D_extra (float): Diffusivity of extra-cellular compartment (m^2/s).
        f_intra (float): Volume fraction of intra-cellular compartment (0..1).
        exchange_time (float): Mean residence time in intra-cellular compartment (s).
    """
    
    def __call__(self, acquisition_scheme, params):
        """
        Compute signal for Karger model.
        
        Args:
            acquisition_scheme: STEAcquisitionScheme-like object. 
                                MUST have .bvalues and .mixing_time.
            params: Dictionary/Pytree containing:
                - D_intra
                - D_extra
                - f_intra
                - exchange_time
                
        Returns:
            Signal estimation (N,)
        """
        D_intra = params['D_intra']
        D_extra = params['D_extra']
        f_intra = params['f_intra']
        tau_i = params['exchange_time']
        
        # Derived parameters
        f_extra = 1.0 - f_intra
        
        # Exchange rates
        # k_ie: rate from Intra to Extra = 1 / tau_i
        # Balance: f_i * k_ie = f_e * k_ei
        # k_ei: rate from Extra to Intra = k_ie * f_i / f_e
        
        k_ie = 1.0 / (tau_i + 1e-9) # Avoid div by zero
        k_ei = k_ie * f_intra / (f_extra + 1e-9)
        
        bvals = acquisition_scheme.bvalues
        # Mixing time (TM) determines the duration of exchange
        # Usually TM in acquisition scheme
        TM = acquisition_scheme.mixing_time
        
        # Pre-allocate signal
        # Since we use matrix exp, we generally have to loop or scan over b-values/TMs if they vary.
        # But if we vectorized, jax.vmap(expm) works on stacked matrices.
        
        # Analytic Solution for 2-Compartment Karger
        # Matrix M = [[-R1, k_ei], 
        #             [k_ie, -R2]]
        # where R1 = b*D_intra/TM + k_ie
        #       R2 = b*D_extra/TM + k_ei
        
        TM_safe = jnp.maximum(TM, 1e-6)
        
        # Decay rates (inclusive of exchange loss)
        R1 = (bvals * D_intra) / TM_safe + k_ie
        R2 = (bvals * D_extra) / TM_safe + k_ei
        
        # Eigenvalues
        # Characteristic eq: (lambda + R1)(lambda + R2) - k_ie*k_ei = 0
        # lambda^2 + (R1+R2)lambda + (R1*R2 - k_ie*k_ei) = 0
        
        tr = -(R1 + R2)
        det = R1*R2 - k_ie*k_ei
        
        # Discriminant
        # delta = sqrt(tr^2 - 4*det) = sqrt((R1+R2)^2 - 4(R1R2 - k1k2))
        #       = sqrt( (R1-R2)^2 + 4*k1*k2 )
        delta = jnp.sqrt((R1 - R2)**2 + 4 * k_ie * k_ei)
        
        lambda_1 = (tr + delta) / 2.0
        lambda_2 = (tr - delta) / 2.0
        
        # Amplitudes (Signal Fractions)
        # Using Spectral Decomposition of the Signal S(TM) = 1^T exp(M*TM) P(0)
        # P1 coeff for exp(lambda_1 * TM)
        # P2 coeff for exp(lambda_2 * TM)
        
        # P1 = ( -lambda_2 - (R1*f_i + R2*f_e) + (k_ie*f_i + k_ei*f_e) ) / (lambda_1 - lambda_2)
        
        R_weighted = R1 * f_intra + R2 * f_extra
        Ex_flux = k_ie * f_intra + k_ei * f_extra
        
        denom = lambda_1 - lambda_2
        safe_denom = jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)
        
        P1 = (-lambda_2 - R_weighted + Ex_flux) / safe_denom
        P2 = 1.0 - P1
        
        # Signal
        # S = P1 * exp(lambda_1 * TM) + P2 * exp(lambda_2 * TM)
        
        signal = P1 * jnp.exp(lambda_1 * TM) + P2 * jnp.exp(lambda_2 * TM)
        
        return signal

if __name__ == "__main__":
    # Self-Test
    print("Testing KargerExchangeModel...")
    
    # Mock Scheme
    import numpy as np
    class MockScheme:
        bvalues = jnp.array([0.0, 1000e6, 2000e6, 3000e6])
        mixing_time = jnp.array([0.1, 0.1, 0.1, 0.1])
        
    scheme = MockScheme()
    
    model = KargerExchangeModel()
    
    # Test 1: No Exchange (tau -> infinity)
    # Should reduce to bi-exponential
    params_no_ex = {
        'D_intra': 2.0e-9,
        'D_extra': 1.0e-9,
        'f_intra': 0.5,
        'exchange_time': 1e9 # Very long
    }
    
    sig_no_ex = model(scheme, params_no_ex)
    print(f"Signal (No Exchange): {sig_no_ex}")
    
    # Analytical bi-exp
    b = scheme.bvalues
    Di = params_no_ex['D_intra']
    De = params_no_ex['D_extra']
    f = params_no_ex['f_intra']
    expected = f * jnp.exp(-b*Di) + (1-f) * jnp.exp(-b*De)
    print(f"Expected (Bi-Exp):    {expected}")
    
    # With the fix, this should now be close
    assert jnp.allclose(sig_no_ex, expected, atol=1e-5)
    print("No-Exchange Limit Passed.")
    
    # Test 2: Fast Exchange (tau -> 0)
    # Should reduce to mono-exponential with D_avg = f*Di + (1-f)*De
    params_fast_ex = {
        'D_intra': 2.0e-9,
        'D_extra': 1.0e-9,
        'f_intra': 0.5,
        'exchange_time': 1e-4 # Very short (0.1ms) relative to TM (100ms)
    }
    
    sig_fast_ex = model(scheme, params_fast_ex)
    print(f"Signal (Fast Exchange): {sig_fast_ex}")
    
    D_avg = f * Di + (1-f) * De
    expected_fast = jnp.exp(-b*D_avg)
    print(f"Expected (Mono-Exp):    {expected_fast}")
    
    # Relax tolerance for fast exchange approximation
    assert jnp.allclose(sig_fast_ex, expected_fast, atol=1e-2)
    print("Fast-Exchange Limit checks out (Qualitative).")
