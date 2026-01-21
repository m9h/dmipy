import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from typing import Tuple, Optional, Union
from functools import partial

from dmipy_jax.core.loss import rician_nll, l1_regularization, prior_loss
from dmipy_jax.utils.spherical_harmonics import sh_basis_real, cart2sphere

class SS3T(eqx.Module):
    """
    Single-Shell 3-Tissue Constrained Spherical Deconvolution (SS3T-CSD) with Prior Support.
    """
    sh_order: int = eqx.field(static=True)
    response_wm_sh: jnp.ndarray
    response_gm: float
    response_csf: float
    lambda_l1: float
    lambda_priors: float
    sigma: float
    
    def __init__(self, 
                 sh_order: int, 
                 response_wm: Union[jnp.ndarray, Tuple[float, float, float]],
                 response_gm: float,
                 response_csf: float,
                 bvalue: float,
                 lambda_l1: float = 1e-2,
                 lambda_priors: float = 1.0,
                 sigma: float = 0.05):
        """
        Args:
            sh_order: Spherical Harmonics order.
            response_wm: Array of SH coeffs or eigenvalues (eval1, eval2, eval3).
            response_gm: GM signal attenuation.
            response_csf: CSF signal attenuation.
            bvalue: Shell b-value.
            lambda_l1: L1 regularization strength for WM SH.
            lambda_priors: Prior loss strength.
            sigma: Rician noise sigma.
        """
        self.sh_order = sh_order
        self.lambda_l1 = lambda_l1
        self.lambda_priors = lambda_priors
        self.sigma = sigma
        self.response_gm = response_gm
        self.response_csf = response_csf
        
        if isinstance(response_wm, (tuple, list)) and len(response_wm) == 3:
            self.response_wm_sh = self._compute_response_sh(jnp.array(response_wm), bvalue, sh_order)
        else:
            self.response_wm_sh = jnp.array(response_wm)

    def _compute_response_sh(self, evals, bval, lmax):
        # Numerical integration for response function
        theta = jnp.linspace(0, jnp.pi, 100)
        phi = jnp.zeros_like(theta)
        
        # S(theta) = exp(-b * (e1*cos^2 + e2*sin^2))
        lambdas = evals
        exponent = -bval * (lambdas[0] * jnp.cos(theta)**2 + lambdas[1] * jnp.sin(theta)**2)
        signal = jnp.exp(exponent)
        
        Y = sh_basis_real(theta, phi, lmax)
        coeffs = jnp.linalg.lstsq(Y, signal, rcond=None)[0]
        
        # Extract zonal harmonics and scale for convolution
        # S_lm = F_lm * R_l * sqrt(4pi/(2l+1))
        # Here 'coeffs' is the fitted response SH.
        # We need to broadcast R_l to all m for each l.
        
        r_l_list = []
        curr = 0
        for l in range(0, lmax + 1, 2):
            n_m = 2 * l + 1
            # Zonal (m=0) index relative to block
            # m range -l..l. m=0 is at index l.
            # E.g. l=0 (size 1) -> idx 0.
            # l=2 (size 5) -> idx 2.
            m0_idx = curr + l
            
            val = coeffs[m0_idx]
            factor = jnp.sqrt(4 * jnp.pi / (2 * l + 1))
            response_val = val * factor
            
            r_l_list.append(jnp.full((n_m,), response_val))
            curr += n_m

        return jnp.concatenate(r_l_list)

    @partial(jax.jit, static_argnums=(0,))
    def fit_voxel(self, 
                  data: jnp.ndarray, 
                  bvecs: jnp.ndarray, 
                  priors: Optional[jnp.ndarray] = None,
                  init_params: Optional[jnp.ndarray] = None):
        """
        Fits a single voxel.
        
        Args:
            data: (N_meas,)
            bvecs: (N_meas, 3)
            priors: Optional (3,) array [P_gm, P_csf, P_wm] fractions.
            init_params: Optional initial guess.
        """
        r, theta, phi = cart2sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])
        Y = sh_basis_real(theta, phi, self.sh_order)
        n_sh = Y.shape[1]
        
        # Initial guess
        if init_params is None:
            # bias towards equal weight if no info
            w_init = jnp.array([-2.0, -2.0]) # approx 0.1
            c_init = jnp.zeros(n_sh)
            c_init = c_init.at[0].set(0.1)
            y0 = jnp.concatenate([w_init, c_init])
        else:
            y0 = init_params

        def loss_fn(params, args):
            data_obs, Y_basis, sigma_val, priors_val = args
            
            p_gm_int, p_csf_int = params[0], params[1]
            c_lm = params[2:]
            
            w_gm = jax.nn.softplus(p_gm_int)
            w_csf = jax.nn.softplus(p_csf_int)
            
            # WM signal
            wm_coeffs_signal = c_lm * self.response_wm_sh
            S_wm = jnp.dot(Y_basis, wm_coeffs_signal)
            
            # Predict
            S_pred = jnp.abs(w_gm * self.response_gm + w_csf * self.response_csf + S_wm)
            
            nll = rician_nll(data_obs, S_pred, sigma_val)
            reg = l1_regularization(c_lm, self.lambda_l1)
            
            loss = nll + reg
            
            # Priors
            if priors_val is not None:
                # Approximate WM fraction from SH c_00
                # WM_frac = c_00 * sqrt(4pi)
                wm_frac = c_lm[0] * jnp.sqrt(4 * jnp.pi)
                
                estimated_fracs = jnp.array([w_gm, w_csf, wm_frac])
                # priors_val expected: [gm, csf, wm]
                p_loss = prior_loss(estimated_fracs, priors_val, self.lambda_priors)
                loss += p_loss
            
            return loss

        solver = optx.BFGS(rtol=1e-4, atol=1e-4)
        
        # Handle priors argument for closure
        # Optimistix args must be valid JAX types. None is tricky if traceable.
        # Use a dummy if None, and a flag? Or just pass None if static?
        # args passed to logic must be arrays.
        # If priors is None, we pass a dummy and check inside? 
        # JIT requires static logic. 
        # Better: fit_voxel uses `None` as Python control flow to pick loss function version?
        # But we want one JIT-able function.
        # Let's assume priors is always passed, but can be NaN or -1 if unused? 
        # Or better: overload/wrapper. 
        # Here: If priors is None, pass dummy.
        
        has_priors = (priors is not None)
        if not has_priors:
             # Dummy compatible shape
             priors_arg = jnp.zeros(3) 
        else:
             priors_arg = priors
             
        # We need check inside loss_fn that handles dummy.
        # Since 'has_priors' is bool, we can't pass it dynamically unless we compile separate versions.
        # We can pass a mask? 
        # Or just support priors=None by separate code path?
        # Simplest: Loss function always takes priors, but weight is 0 if not provided.
        
        # Redefine loss to use masking
        def loss_fn_robust(params, args):
            data_obs, Y_basis, sigma_val, priors_val, prior_strength = args
            
            p_gm_int, p_csf_int = params[0], params[1]
            c_lm = params[2:]
            
            w_gm = jax.nn.softplus(p_gm_int)
            w_csf = jax.nn.softplus(p_csf_int)
            
            wm_coeffs_signal = c_lm * self.response_wm_sh
            S_wm = jnp.dot(Y_basis, wm_coeffs_signal)
            S_pred = jnp.abs(w_gm * self.response_gm + w_csf * self.response_csf + S_wm)
            
            nll = rician_nll(data_obs, S_pred, sigma_val)
            reg = l1_regularization(c_lm, self.lambda_l1)
            
            wm_frac = c_lm[0] * jnp.sqrt(4 * jnp.pi)
            estimated_fracs = jnp.array([w_gm, w_csf, wm_frac])
            
            # Prior loss (only if strength > 0)
            p_loss = prior_loss(estimated_fracs, priors_val, prior_strength)
            
            return nll + reg + p_loss

        # Actual strength
        eff_lambda_priors = self.lambda_priors if has_priors else 0.0
        
        args = (data, Y, self.sigma, priors_arg, eff_lambda_priors)
        
        sol = optx.minimize(
            fn=loss_fn_robust,
            solver=solver,
            y0=y0,
            args=args,
            max_steps=500,
            throw=False
        )
        
        p = sol.value
        w_gm = jax.nn.softplus(p[0])
        w_csf = jax.nn.softplus(p[1])
        c_lm = p[2:]
        
        return jnp.concatenate([jnp.array([w_gm, w_csf]), c_lm]), sol.result

    def predict(self, params, bvecs):
        w_gm = params[0]
        w_csf = params[1]
        c_lm = params[2:]
        r, theta, phi = cart2sphere(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])
        Y = sh_basis_real(theta, phi, self.sh_order)
        wm_coeffs = c_lm * self.response_wm_sh
        S_wm = jnp.dot(Y, wm_coeffs)
        return w_gm * self.response_gm + w_csf * self.response_csf + S_wm
