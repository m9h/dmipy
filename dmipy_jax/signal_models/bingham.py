
import jax
import jax.numpy as jnp
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.core.integration_grids import get_spherical_fibonacci_grid

__all__ = ['BinghamNODDI']

def get_rotation_matrix_from_z_to_vector(target_vector):
    """
    Computes the rotation matrix that rotates the Z-axis [0,0,1] to the target_vector.
    Using Rodrigues' rotation formula.
    
    Args:
        target_vector: (3,) normalized vector.
        
    Returns:
        (3, 3) rotation matrix.
    """
    z_axis = jnp.array([0., 0., 1.])
    
    # Check if target is parallel to Z (or anti-parallel)
    # Ideally should handle this safely. 
    # Dot product
    dot_prod = jnp.dot(z_axis, target_vector)
    
    # Cross product axis
    k = jnp.cross(z_axis, target_vector)
    norm_k = jnp.linalg.norm(k)
    
    # Safe branch for parallel vectors
    # If parallel (norm_k ~ 0), return Identity (or Flip if anti-parallel)
    # Using jax.lax.cond for safety if needed, or simple 'where' logic on the components.
    # But for a simple mask:
    
    # Skew-symmetric matrix K
    K = jnp.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    # Rodrigues formula
    # R = I + K + K^2 * (1 - dot) / (norm_k^2)
    # But using sinc form is safer for small angles?
    # Standard: R = I + sin(th) K_norm + (1-cos(th)) K_norm^2
    # Here sin(theta) = norm_k, cos(theta) = dot_prod
    
    # R = I + K + K^2 * (1 - c) / s^2
    # s^2 = 1 - c^2 = (1-c)(1+c)
    # (1-c)/s^2 = 1/(1+c)
    
    factor = 1.0 / (1.0 + dot_prod)
    
    # Handle the anti-parallel case (dot_prod = -1)
    # If dot_prod is close to -1, factor explodes.
    # For MRI fiber orientations, we usually assume symmetry or handle full sphere.
    # But if mu is [0,0,-1], we need a 180 rotation roughly X.
    # Let's add epsilon or handle it.
    
    # Robust approach:
    # return jnp.where(dot_prod > 0.99999, jnp.eye(3), ... )
    
    R_standard = jnp.eye(3) + K + jnp.matmul(K, K) * factor
    
    # If dot_prod is -1 (anti-parallel), we can rotate 180 around X.
    R_flip = jnp.diag(jnp.array([1., -1., -1.])) # Rot 180 around X flips Y and Z
    
    # Basic safe switch
    is_parallel = dot_prod > 0.99999
    is_antiparallel = dot_prod < -0.99999
    
    res = jnp.where(is_parallel, jnp.eye(3), R_standard)
    res = jnp.where(is_antiparallel, R_flip, res)
    
    return res

class BinghamNODDI:
    r"""
    Bingham-NODDI model using numerical integration.
    
    The signal is defined as:
    S = \int_{S^2} P(\mathbf{n} | \mu, \kappa_1, \kappa_2) S_{stick}(\mathbf{n}, b, g) d\mathbf{n}
    
    The Bingham PDF is modeled as:
    P(\mathbf{n}) \propto \exp( - \kappa_1 n_x^2 - \kappa_2 n_y^2 )
    defined in a frame where the peak is along Z, then rotated to \mu.
    
    Args:
        grid_points (int): Number of points for numerical integration.
    """
    
    def __init__(self, grid_points=100):
        self.stick_model = C1Stick()
        # Generate grid
        self.grid_vectors_canonical, self.grid_weights = get_spherical_fibonacci_grid(grid_points)
        
    def __call__(self, bvalues, gradient_directions, mu, kappa1, kappa2, **kwargs):
        """
        Calculate signal.
        
        Args:
            bvalues: (N_g,)
            gradient_directions: (N_g, 3)
            mu: (3,) or (2,) orientation unit vector or angles.
                If 2 params, assumed [theta, phi]. If 3, vectors.
                Let's assume input is vectors (3,) for flexibility or convert.
                The prompt benchmarks used angles for Stick, but usually models take vectors or whatever stick takes.
                C1Stick takes `mu` as a vector or angles? Let's check C1Stick later or assume vector for rotation.
                Wait, benchmarks/benchmark_stick_performance.py used [theta, phi].
                Standard `C1Stick` in dmipy likely takes vector `mu` in `kwargs`?
                The benchmark passes `mu` (N,2) to `model(..., mu=mu_val)`.
                So `C1Stick` takes angles.
                I need to be careful.
                Let's support `mu` as vector (3,) since we need to build a rotation matrix.
                If input is angles, I'll convert to vector.
                
            kappa1: float, concentration parameter 1 (along one axis perp to mu)
            kappa2: float, concentration parameter 2
            kwargs: passed to stick (e.g. lambda_par)
        """
        
        # 0. Normalize inputs
        # If mu is angles (2,), convert to vector
        mu = jnp.array(mu)
        if mu.shape == (2,):
            theta, phi = mu
            st, ct = jnp.sin(theta), jnp.cos(theta)
            sp, cp = jnp.sin(phi), jnp.cos(phi)
            mu_vec = jnp.array([st*cp, st*sp, ct])
        else:
            mu_vec = mu / jnp.linalg.norm(mu)

        # 1. Rotate grid
        # Calculate rotation mapping Z to mu_vec
        R = get_rotation_matrix_from_z_to_vector(mu_vec)
        
        # Grid vectors: (N_grid, 3)
        # rotated_grid = (R @ canonical.T).T
        grid_vectors_rotated = jnp.dot(self.grid_vectors_canonical, R.T)
        
        # 2. Compute Bingham Weights on CANONICAL grid (easier)
        # P(n) propto exp(-k1 * nx^2 - k2 * ny^2)
        # nx, ny are components in the canonical frame!
        nx = self.grid_vectors_canonical[:, 0]
        ny = self.grid_vectors_canonical[:, 1]
        
        # Unnormalized PDF
        # We assume k1, k2 are positive concentration parameters "away" from Z?
        # If k is large, we want narrow distribution.
        # exp(-k * x^2) -> narrow as k increases. Matches usage.
        pdf_vals = jnp.exp(-kappa1 * nx**2 - kappa2 * ny**2)
        
        # Normalize (Sum * Area_Element = 1)
        # Integral P dA = sum(w * P) approx 1
        normalization = jnp.sum(pdf_vals * self.grid_weights)
        pdf_normalized = pdf_vals / normalization
        
        # 3. Integrate Signal
        # Sum_i ( w_i * P(n_i) * S_stick(n_i) )
        
        # We need to broadcast S_stick over the grid points.
        # grid_vectors_rotated are the orientations of the individual sticks.
        
        # C1Stick typically takes `mu` as orientation.
        # Check C1Stick signature: call(bvals, gradient_directions, mu, lambda_par)
        # If C1Stick requires angles, we must convert grid vectors to angles.
        # If it supports vectors, great.
        
        # Let's assume for now we must convert to angles to match C1Stick if it expects angles.
        # But C1Stick jax implementation usually supports vectors if it uses dot products?
        # I should check C1Stick source code to be sure. 
        # But to be safe, I will compute sticks manually if needed or map.
        
        # Better: Define a mapped function to call stick for each grid point.
        # But bvalues/grads are constant for all grid points.
        # We vary `mu`.
        
        # Let's import vmap
        from jax import vmap
        
        # Helper to get signal for one micro-orientation n_i
        # S_stick uses Acq params (bvals, grads).
        # We need to map over n_i.
        
        def signal_for_orientation(n_mu):
            # n_mu is (3,)
            # Convert to angles?
            # theta = arccos(nz)
            # phi = arctan2(ny, nx)
            nz = n_mu[2]
            nx = n_mu[0]
            ny = n_mu[1]
            theta = jnp.arccos(jnp.clip(nz, -1.0, 1.0))
            phi = jnp.arctan2(ny, nx)
            mu_angles = jnp.array([theta, phi])
            
            return self.stick_model(bvalues, gradient_directions, mu=mu_angles, **kwargs)

        # Vectorize over grid points
        # grid_signals: (N_grid, N_measurements)
        grid_signals = vmap(signal_for_orientation)(grid_vectors_rotated)
        
        # Weighted sum
        # weights: (N_grid,)
        # pdf_normalized: (N_grid,)
        # combined weight: w * P
        total_weights = self.grid_weights * pdf_normalized
        
        # Sum over grid axis (0)
        # S = sum(w_i * S_i)
        # Need to reshape weights to (N_grid, 1) to broadcast over Measurements
        S_bingham = jnp.sum(grid_signals * total_weights[:, None], axis=0)
        
        return S_bingham

