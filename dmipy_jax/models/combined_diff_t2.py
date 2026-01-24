
import jax.numpy as jnp
import equinox as eqx
from dmipy_jax.signal_models.stick import Stick
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.gaussian_models import Ball

class CombinedStandardModel(eqx.Module):
    """
    Combined Diffusion-Relaxometry Standard Model.
    
    Components:
        1. Intra-neurite (Stick): f_in, D_in, T2_in, orientation
        2. Extra-neurite (Zeppelin): f_ex, D_ex_par, D_ex_perp, T2_ex, orientation
        3. CSF (Ball): f_csf, D_csf, T2_csf
        
    Signal Equation:
        S(b, n, TE) = f_in * S_stick(b, n) * exp(-TE/T2_in) + 
                      f_ex * S_zeppelin(b, n) * exp(-TE/T2_ex) + 
                      f_csf * S_ball(b) * exp(-TE/T2_csf)
                      
    Note: Signal is normalized such that f_in + f_ex + f_csf = 1 (if S0 is factored out).
    """
    
    stick: Stick = eqx.field(static=True)
    zeppelin: Zeppelin = eqx.field(static=True)
    ball: Ball = eqx.field(static=True)
    
    def __init__(self):
        self.stick = Stick()
        self.zeppelin = Zeppelin()
        self.ball = Ball()
        
    def __call__(self, parameters, acquisition_scheme):
        """
        Args:
            parameters: Dictionary.
            acquisition_scheme: Must have 'TE' attribute.
        """
        bvals = acquisition_scheme.bvalues
        bvecs = acquisition_scheme.gradient_directions
        te = getattr(acquisition_scheme, 'TE', None)
        if te is None:
            raise ValueError("Acquisition scheme must have 'TE' attribute.")
        
        # Unpack parameters
        f_in = parameters['f_in']
        f_ex = parameters['f_ex']
        f_csf = parameters['f_csf']
        
        # Stick (Intra)
        S_stick = self.stick(bvals, bvecs, lambda_par=parameters['D_in'], mu=parameters['mu'])
        
        # Zeppelin (Extra)
        S_zeppelin = self.zeppelin(bvals, bvecs, 
                                   lambda_par=parameters['D_ex_par'], 
                                   lambda_perp=parameters['D_ex_perp'], 
                                   mu=parameters['mu'])
        
        # Ball (CSF)
        S_ball = self.ball(bvals, None, lambda_iso=parameters['D_csf']) 
        
        # Relaxometry Attenuation
        # Ensure T2 are floats to avoid division by zero or int issues
        # TE is in ms, T2 in ms.
        E_in = jnp.exp(-te / parameters['T2_in'])
        E_ex = jnp.exp(-te / parameters['T2_ex'])
        E_csf = jnp.exp(-te / parameters['T2_csf'])
        
        # Combine
        S_total = (f_in * S_stick * E_in) + \
                  (f_ex * S_zeppelin * E_ex) + \
                  (f_csf * S_ball * E_csf)
                  
        return S_total
