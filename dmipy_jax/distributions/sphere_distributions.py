import jax
import jax.numpy as jnp
import equinox as eqx

__all__ = ['SD1Watson']

class SD1Watson(eqx.Module):
    r"""
    Watson distribution on the unit sphere.
    
    PDF: f(n; mu, kappa) = (1/Z) * exp(kappa * (n . mu)^2)
    
    This implementation uses a fixed spherical grid for numerical integration and normalization.
    """
    
    mu: jnp.ndarray
    kappa: float
    _grid_vectors: jnp.ndarray 
    
    parameter_names = ['mu', 'kappa']
    parameter_cardinality = {'mu': 2, 'kappa': 1} # mu is (theta, phi) -> 2 scalars, but we might input vector? 
    # Usually dmipy models take parameters as scalars or arrays. 
    # mu is orientation. In dmipy, orientation is often (theta, phi) or (x,y,z).
    # The Stick/Zeppelin models take 'mu' as 2-element array (theta, phi) usually?
    # Let's check c_noddi.py: "mu = jnp.array([theta, phi])".
    # So we stick to that convention.
    
    parameter_ranges = {
        'mu': [(0.0, jnp.pi), (-jnp.pi, jnp.pi)], 
        'kappa': (0.0, 32.0) # Reasonable range for dispersion
    }

    def __init__(self, mu=None, kappa=None, grid_size=200):
        self.mu = mu
        self.kappa = kappa
        self._grid_vectors = self._fibonacci_sphere(grid_size)
    
    def _fibonacci_sphere(self, samples=200):
        """Generates points on a sphere using Fibonacci spiral."""
        points = []
        phi = jnp.pi * (3. - jnp.sqrt(5.))  # golden angle in radians

        # We can't use simple python loop for JIT compatibility if we want this dynamic, 
        # but grid_size is static in __init__.
        # So we can use numpy/python logic here.
        import numpy as np
        
        y = np.linspace(1, -1, samples)
        radius = np.sqrt(1 - y * y)
        theta = phi * np.arange(samples)

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        return jnp.array(np.stack([x, y, z], axis=1))

    def __call__(self, **kwargs):
        """
        Returns grid vectors and their weights (normalized).
        
        Args:
            mu: (2,) array [theta, phi]
            kappa: scalar
            
        Returns:
            vectors: (N, 2) array of [theta, phi] for the grid points.
            weights: (N,) array of weights summing to 1.
        """
        # 1. Parse Inputs
        mu_angles = kwargs.get('mu', self.mu) # [theta, phi]
        kappa = kwargs.get('kappa', self.kappa)
        
        # Convert mu to vector
        mu_vec = self._spherical_to_cartesian(mu_angles)
        
        # 2. Compute Probabilities (Unnormalized)
        # grid vectors are cartesian (N, 3)
        # dot product: (N, 3) @ (3,) -> (N,)
        dot_prod = self._grid_vectors @ mu_vec
        log_unnorm_prob = kappa * (dot_prod ** 2)
        
        # stabilize exp
        # max_val = jnp.max(log_unnorm_prob)
        # unnorm_prob = jnp.exp(log_unnorm_prob - max_val)
        unnorm_prob = jnp.exp(log_unnorm_prob)
        
        # 3. Normalize
        Z = jnp.sum(unnorm_prob)
        weights = unnorm_prob / Z
        
        # 4. Return Output
        # DistributedModel expects "domain". For spherical models, the "domain" 
        # is the set of parameters fed to the sub-model.
        # The sub-model (Stick/Zeppelin) expects 'mu' as angles [theta, phi].
        # So we must convert our grid vectors back to angles.
        
        grid_angles = self._cartesian_to_spherical(self._grid_vectors)
        
        return grid_angles, weights

    def _spherical_to_cartesian(self, angles):
        theta, phi = angles[0], angles[1]
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        return jnp.array([x, y, z])

    def _cartesian_to_spherical(self, vectors):
        # vectors: (N, 3)
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        r = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))
        phi = jnp.arctan2(y, x)
        return jnp.stack([theta, phi], axis=1)
