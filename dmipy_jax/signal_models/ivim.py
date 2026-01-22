
import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx
from typing import Any

@jit
def g_ivim(bvals, D_tissue, D_pseudo, f):
    """
    Computes signal for the IVIM model (Bi-exponential decay).
    
    S/S0 = (1 - f) * exp(-b * D_tissue) + f * exp(-b * D_pseudo)
    
    Args:
        bvals: (N,) array of b-values.
        D_tissue: Diffusion coefficient of tissue (slow).
        D_pseudo: Pseudo-diffusion coefficient of perfusion (fast).
        f: Perfusion fraction (0 to 1).
    """
    # Standard IVIM formulation
    # Note: Some formulations use D* for pseudo. Here D_pseudo.
    
    term_tissue = (1 - f) * jnp.exp(-bvals * D_tissue)
    term_pseudo = f * jnp.exp(-bvals * D_pseudo)
    
    return term_tissue + term_pseudo


class IVIM(eqx.Module):
    r"""
    Intravoxel Incoherent Motion (IVIM) model [1]_.
    
    Parameters
    ----------
    D_tissue : float
        Tissue diffusion coefficient (m^2/s). Typically ~ 0.7e-9 to 1.0e-9.
    D_pseudo : float
        Pseudo-diffusion coefficient (m^2/s). Typically >> D_tissue.
    f : float
        Perfusion volume fraction.
        
    References
    ----------
    .. [1] Le Bihan, Denis, et al. "Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging." 
           Radiology 168.2 (1988): 497-505.
    """
    
    D_tissue: Any = None
    D_pseudo: Any = None
    f: Any = None

    parameter_names = ('D_tissue', 'D_pseudo', 'f')
    parameter_cardinality = {
        'D_tissue': 1, 
        'D_pseudo': 1, 
        'f': 1
    }
    # Typical bounds for optimization
    parameter_ranges = {
        'D_tissue': (0.0, 3e-9),
        'D_pseudo': (0.0, 100e-9), # D* is often much larger
        'f': (0.0, 1.0)
    }

    def __init__(self, D_tissue=None, D_pseudo=None, f=None):
        self.D_tissue = D_tissue
        self.D_pseudo = D_pseudo
        self.f = f

    def __call__(self, bvals, gradient_directions=None, **kwargs):
        # IVIM is usually isotropic in basic formulation, so gradient_directions are ignored
        # but kept for API consistency.
        
        D_tissue = kwargs.get('D_tissue', self.D_tissue)
        D_pseudo = kwargs.get('D_pseudo', self.D_pseudo)
        f = kwargs.get('f', self.f)
        
        return g_ivim(bvals, D_tissue, D_pseudo, f)
