
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G1Ball

__all__ = ['BallStick']

class BallStick:
    r"""
    Ball and Stick Model.
    
    Composition:
    1. Intra-cellular (f_stick): Stick (C1Stick) - Anisotropic, zero radius.
    2. Extra-cellular / Isotropic (1 - f_stick): Ball (G1Ball) - Isotropic.
    
    Parameters typically fitted:
    - theta, phi: Orientation of the stick.
    - f_stick: Volume fraction of the stick compartment.
    
    Fixed Parameters:
    - diffusivity: Diffusivity for both stick (axial) and ball (isotropic).
                   Default is 1.7e-9 m^2/s.
    """
    
    def __init__(self, diffusivity=1.7e-9):
        self.diffusivity = diffusivity
        
        self.stick = C1Stick()
        self.ball = G1Ball()
        
    def __call__(self, params, acquisition):
        """
        Calculate signal attenuation.
        
        Args:
            params (jnp.ndarray): 1D array of parameters [theta, phi, f_stick].
            acquisition (JaxAcquisition): Acquisition object containing bvalues, gradient_directions.
            
        Returns:
            jnp.ndarray: Signal attenuation.
        """
        theta = params[0]
        phi = params[1]
        mu = jnp.array([theta, phi])
        
        f_stick = params[2]
        f_ball = 1.0 - f_stick
        
        # Signals
        # Note: Handling btensors if acquisition supports it is handled by sub-models if they support it.
        # Here we pass kwargs if needed, but sub-models C1Stick/G1Ball in dmipy_jax usually handle bvals/grads directly or via tensor interface if generalized.
        # My C1Stick implementation in dmipy_jax checks for btensors in acquisition object if passed?
        # In c_noddi.py, I passed:
        # S_stick = self.stick(bvals=..., gradient_directions=..., mu=..., lambda_par=...)
        # I should assume standard calling convention.
        
        # Check if acquisition has btensors
        btensors = getattr(acquisition, 'btensors', None)
        
        S_stick = self.stick(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=self.diffusivity,
            btensors=btensors
        )
        
        S_ball = self.ball(
            bvals=acquisition.bvalues,
            lambda_iso=self.diffusivity,
            btensors=btensors
        )
        
        S_total = f_stick * S_stick + f_ball * S_ball
        
        return S_total
