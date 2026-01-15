
import jax
import jax.numpy as jnp
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G1Ball, G2Zeppelin

def add_t2_decay(signal, te, t2):
    """
    Modulates the signal with T2 decay: S_new = S_old * exp(-TE / T2).
    
    Args:
        signal (jnp.ndarray): The diffusion-weighted signal (without T2 decay).
        te (jnp.ndarray): Echo times in seconds. Same shape as signal or broadcastable.
        t2 (float or jnp.ndarray): T2 relaxation time in seconds.
        
    Returns:
        jnp.ndarray: Signal with T2 decay appled.
    """
    # Ensure positive T2 to avoid exploding signal (optional but good practice)
    # The caller is responsible for softplus on parameters, but we can check or clip here if needed.
    # For now, assume t2 is already processed/positive.
    
    decay = jnp.exp(-te / t2)
    return signal * decay

class MTE_NODDI:
    r"""
    Multi-Echo NODDI (MTE-NODDI) model implementation in JAX.
    
    This model extends the standard NODDI model by accounting for compartment-specific
    T2 relaxation times, which is enabled by multi-TE acquisitions.
    
    The signal model is defined as:
    S = f_iso * exp(-TE/T2_iso) * S_iso 
      + f_ic * exp(-TE/T2_ic) * S_ic 
      + f_ec * exp(-TE/T2_ec) * S_ec
      
    Where:
    - S_iso is isotropic diffusion (G1Ball)
    - S_ic is intra-cellular diffusion (C1Stick)
    - S_ec is extra-cellular diffusion (G2Zeppelin with tortuosity constraint)
    
    The volume fractions are parameterized as:
    - f_iso: Global isotropic volume fraction
    - f_ic (global) = (1 - f_iso) * v_ic
    - f_ec (global) = (1 - f_iso) * (1 - v_ic)
    
    Where v_ic is the intra-cellular volume fraction of the non-isotropic tissue compartment.
    The tortuosity constraint is applied to S_ec: d_perp = d_par * (1 - v_ic).
    """
    
    parameter_names = [
        'f_iso', 'v_ic', 
        't2_iso', 't2_ic', 't2_ec', 
        'mu', 'd_par', 'd_iso'
    ]
    # Cardinality: mu is 2 (theta, phi), others are 1.
    parameter_cardinality = {
        'f_iso': 1, 'v_ic': 1,
        't2_iso': 1, 't2_ic': 1, 't2_ec': 1,
        'mu': 2, 'd_par': 1, 'd_iso': 1
    }
    
    def __init__(self, f_iso=None, v_ic=None, t2_iso=None, t2_ic=None, t2_ec=None, 
                 mu=None, d_par=1.7e-9, d_iso=3.0e-9):
        self.f_iso = f_iso
        self.v_ic = v_ic
        self.t2_iso = t2_iso
        self.t2_ic = t2_ic
        self.t2_ec = t2_ec
        self.mu = mu
        self.d_par = d_par
        self.d_iso = d_iso
        
        # Sub-models
        self.stick = C1Stick()
        self.zeppelin = G2Zeppelin()
        self.ball = G1Ball()

    def __call__(self, acquisition, **kwargs):
        """
        Predicts the signal for the given acquisition parameters.
        
        Args:
            acquisition (JaxAcquisition): The acquisition object containing bvalues, 
                                          gradient_directions, and echo_time.
            **kwargs: Model parameters (f_iso, v_ic, t2_iso, t2_ic, t2_ec, mu, d_par, d_iso).
            
        Returns:
            jnp.ndarray: The predicted signal.
        """
        # 1. Unpack parameters with defaults
        f_iso = kwargs.get('f_iso', self.f_iso)
        v_ic = kwargs.get('v_ic', self.v_ic)
        
        t2_iso = kwargs.get('t2_iso', self.t2_iso)
        t2_ic = kwargs.get('t2_ic', self.t2_ic)
        t2_ec = kwargs.get('t2_ec', self.t2_ec)
        
        mu = kwargs.get('mu', self.mu)
        d_par = kwargs.get('d_par', self.d_par)
        d_iso = kwargs.get('d_iso', self.d_iso)
        
        # 2. Enforce constraints (Softplus for T2 and D, Sigmoid for fractions if fitting unconstrained)
        # Assuming the fitting routine handles parameter transformation or passing constrained values here.
        # But for T2, the prompt explicitly asked: "Use jax.nn.softplus to enforce positivity constraints on T2"
        # I will apply softplus ONLY if the user provides raw unconstrained parameters? 
        # Usually, the model __call__ expects physical parameters. The mapping from unconstrained->physical
        # happens in the objective function or a wrapper. 
        # HOWEVER, the prompt said: "Optimization: This adds 3 new parameters... Use jax.nn.softplus...".
        # This implies the model might receive unconstrained parameters or I should provide a method to transform.
        # BUT standard practice in Dmipy/JAX is that __call__ receives PHYSICAL parameters.
        # I will stick to PHYSICAL parameters in __call__. The optimizer usage of softplus belongs in the fitting module.
        # Wait, if I'm just implementing the model class, I should assume valid inputs.
        # I will leave the inputs as-is, assuming they are physical. 
        # *Correction*: The user prompt might imply I should put the softplus inside if I want safe behavior.
        # Let's assume standard behavior: inputs to __call__ are physical.
        
        # 3. Calculate Derived Fractions
        f_ic_global = (1 - f_iso) * v_ic
        f_ec_global = (1 - f_iso) * (1 - v_ic)
        
        # 4. Calculate Signals (Diffusion only)
        
        # Intra-cellular (Stick)
        S_stick = self.stick(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=d_par
        )
        
        # Extra-cellular (Zeppelin with Tortuosity)
        # Tortuosity: d_perp = d_par * (1 - v_ic)
        d_perp = d_par * (1 - v_ic)
        S_zeppelin = self.zeppelin(
            bvals=acquisition.bvalues,
            gradient_directions=acquisition.gradient_directions,
            mu=mu,
            lambda_par=d_par,
            lambda_perp=d_perp
        )
        
        # Isotropic (Ball)
        S_ball = self.ball(
            bvals=acquisition.bvalues,
            lambda_iso=d_iso
        )
        
        # 5. Apply T2 Decay
        TE = acquisition.echo_time
        if TE is None:
            raise ValueError("MTE_NODDI requires 'echo_time' in the acquisition object.")
            
        S_stick_T2 = add_t2_decay(S_stick, TE, t2_ic)
        S_zeppelin_T2 = add_t2_decay(S_zeppelin, TE, t2_ec)
        S_ball_T2 = add_t2_decay(S_ball, TE, t2_iso)
        
        # 6. Composite Signal
        S_total = (f_ic_global * S_stick_T2 + 
                   f_ec_global * S_zeppelin_T2 + 
                   f_iso * S_ball_T2)
                   
        return S_total
