import equinox as eqx
import jax
import jax.numpy as jnp
from dmipy_jax.basis.spherical_harmonics import sh_basis

class SphericalHarmonicsFit(eqx.Module):
    basis_matrix: jax.Array
    pinv_basis: jax.Array
    order: int = eqx.field(static=True)
    
    def __init__(self, gradients, order):
        """
        Initialize SH Fitter.
        
        Args:
            gradients: (N, 3) array of gradient directions.
            order: Maximum SH order (even).
        """
        self.order = order
        self.basis_matrix = sh_basis(gradients, order)
        # Compute Moore-Penrose Pseudoinverse for least-squares fitting
        # (N_coeffs, N_dirs)
        self.pinv_basis = jnp.linalg.pinv(self.basis_matrix)
        
    def __call__(self, signal):
        """
        Fit SH coefficients to signal.
        
        Args:
            signal: (N_dirs,) signal array.
            
        Returns:
            coeffs: (N_coeffs,) SH coefficients.
        """
        # coeffs = pinv @ signal
        return jnp.dot(self.pinv_basis, signal)
