import jax
import jax.numpy as jnp
from jax.scipy import stats
from jax import jit

__all__ = ['DD1Gamma']

class DD1Gamma:
    r"""A Gamma distribution of cylinder diameter for given alpha and beta
    parameters.
    
    Parameters
    ----------
    alpha : float,
        shape of the gamma distribution.
    beta : float,
        scale of the gamma distrubution.
    """
    
    parameter_names = ['alpha', 'beta']
    parameter_cardinality = {'alpha': 1, 'beta': 1}
    parameter_ranges = {
        'alpha': (0.1, 30.),
        'beta': (1e-3, 2.)
    }

    def __init__(self, alpha=None, beta=None, Nsteps=50):
        self.alpha = alpha
        self.beta = beta
        self.Nsteps = Nsteps

    def __call__(self, **kwargs):
        """
        Returns the sampling grid (radii) and their probability density (P_radii).
        
        Returns
        -------
        radii : (Nsteps,) array
            radii values sampled.
        P_radii : (Nsteps,) array
            Normalized probability density at those radii.
        """
        alpha = kwargs.get('alpha', self.alpha)
        beta = kwargs.get('beta', self.beta)
        
        # Calculate grid dynamically based on mean/std to cover most of the distribution
        # mean = alpha * beta
        # std = sqrt(alpha) * beta
        # range: 0.01e-6 to mean + 6*std (approximation)
        
        mean = alpha * beta
        std = jnp.sqrt(alpha) * beta
        
        limit = mean + 6 * std
        
        # Avoid zeros or extremely small values
        start = 0.1e-6 # 0.1 microns
        end = jnp.maximum(limit, start + 1e-6)
        
        # Fixed grid size (Nsteps)
        radii = jnp.linspace(start, end, self.Nsteps)
        
        # Calculate PDF
        # jax.scipy.stats.gamma.pdf(x, a, scale=1) -> standard gamma
        # Gamma(x; alpha, beta) = x^(alpha-1) * exp(-x/beta) / (beta^alpha * Gamma(alpha))
        # This matches scipy.stats.gamma(a=alpha, scale=beta)
        # JAX gamma.pdf definition: gamma.pdf(x, a) is for standard gamma (beta=1?)
        # Let's check JAX docs logic: pdf(x, a) = x**(a-1) * exp(-x) / gamma(a).
        # So for non-standard scale beta:
        # pdf(x, alpha, beta) = (1/beta) * standard_pdf(x/beta, alpha)
        
        pdf_vals = stats.gamma.pdf(radii / beta, alpha) / beta
        
        # Normalize
        # Area = (end - start) / (Nsteps - 1) * sum(pdf) approx
        # We handle normalization explicitly to ensure sum(weights) = 1 for integration?
        # Or return true PDF? distributed_model usually does trapezoidal integration.
        # Let's return true PDF vals.
        
        # Apply normalization func if needed (e.g. surface/volume weighted)
        # Legacy had 'normalization' arg. Default was 'standard' (unity).
        # We'll stick to standard PDF for now.
        
        return radii, pdf_vals
