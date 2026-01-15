
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G1Ball, G2Zeppelin

__all__ = ['CNODDI']

class CNODDI:
    r"""
    Constrained NODDI (C-NODDI) Model.
    
    This model enforces a tortuosity constraint on the extra-cellular compartment:
    D_extra_perp = D_extra_par * (1 - f_ic)
    
    It assumes a Single-Fascicle (non-dispersed) configuration in this implementation,
    composed of:
    1. Intra-cellular (f_ic): Stick (C1Stick)
    2. Extra-cellular (1 - f_ic - f_iso): Zeppelin (G2Zeppelin) with constrained perpendicular diffusivity.
    3. Isotropic (f_iso): Ball (G1Ball) representing CSF.
    
    Parameters typically fitted:
    - theta, phi: Orientation
    - f_stick: Intra-cellular volume fraction (f_ic)
    - f_iso: Isotropic volume fraction
    
    Fixed Parameters (typically):
    - diffusivity_par: 1.7e-9 m^2/s (or similar, e.g., 3.0e-9)
    - diffusivity_iso: 3.0e-9 m^2/s
    """
    
    def __init__(self, diffusivity_par=1.7e-9, diffusivity_iso=3.0e-9):
        self.diffusivity_par = diffusivity_par
        self.diffusivity_iso = diffusivity_iso
        
        self.stick = C1Stick()
        self.zeppelin = G2Zeppelin()
        self.ball = G1Ball()
        
    def __call__(self, params, acquisition):
        """
        Calculate signal attenuation.
        
        Args:
            params (jnp.ndarray): 1D array of parameters [theta, phi, f_stick, f_iso].
            acquisition (JaxAcquisition): Acquisition object containing bvalues, gradient_directions.
            
        Returns:
            jnp.ndarray: Signal attenuation.
        """
        theta = params[0]
        phi = params[1]
        mu = jnp.array([theta, phi])
        
        f_stick = params[2]
        f_iso = params[3]
        
        # Enforce physical constraints / basic bounds handled by optimizer, 
        # but here we ensure consistency of fractions for calculation.
        f_extra = 1.0 - f_stick - f_iso
        
        # Tortuosity Constraint:
        # D_extra_perp = D_extra_par * (1 - f_ic)
        # Here D_extra_par is assumed to be the same as D_intra_par (diffusivity_par)
        lambda_par = self.diffusivity_par
        lambda_perp = lambda_par * (1.0 - f_stick)
        
        # Signals
        S_stick = self.stick(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=lambda_par
        )
        
        S_zeppelin = self.zeppelin(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=lambda_par,
            lambda_perp=lambda_perp
        )
        
        S_ball = self.ball(
            bvals=acquisition.bvalues,
            lambda_iso=self.diffusivity_iso
        )
        
        # Composition
        # Note: We should probably ensure f_extra is non-negative, but for now we trust the inputs 
        # or let the physics break if invalid. 
        # Actually, if f_stick + f_iso > 1, f_extra becomes negative.
        # This is strictly a signal generator, optimization constraints should handle valid fractions.
        
        S_total = f_stick * S_stick + f_extra * S_zeppelin + f_iso * S_ball
        
        return S_total
