import jax
import jax.numpy as jnp
import equinox as eqx
import e3nn_jax as e3nn
from typing import Optional, Callable

class BucklingLayer(eqx.Module):
    """
    Simulates cortical folding (buckling) using a spectral growth model on a spherical surface.
    
    The surface is represented by a scalar radius function R(theta, phi) decomposed into 
    Sherical Harmonics (SH).
    
    Growth is applied as a spectral filter that amplifies high-frequency modes (l > l0).
    """
    
    lmax: int = eqx.field(static=True)
    l0: float # Cutoff frequency (soft)
    k: float  # Steepness of transition
    
    def __init__(self, lmax: int = 8, l0: float = 2.0, k: float = 5.0):
        self.lmax = lmax
        self.l0 = l0
        self.k = k
        
    def growth_operator(self, coeffs: jax.Array, g: float) -> jax.Array:
        """
        Applies spectral growth to SH coefficients.
        
        Args:
            coeffs: SH coefficients of shape (K,), where K = (lmax+1)^2.
            g: Growth ratio parameter (scalar).
            
        Returns:
            Grown coefficients.
        """
        # Generate indices l for each coefficient
        # e3nn uses ordering: 0, 1,-1,1, 2,-2,2,-1,1?? No, standard is usually l^2 + l + m
        # e3nn Irreps: "0e + 1o + 2e + ..."
        # Usually it is stacked by l.
        
        ls = []
        for l in range(self.lmax + 1):
            ls.extend([l] * (2 * l + 1))
        ls = jnp.array(ls)
        
        # Spectral filter function k(l)
        # Sigmoid-like transition: 1 / (1 + exp(-k * (l - l0)))
        # We want k(l) to be close to 0 for l < l0 and 1 for l > l0
        
        filter_val = 1.0 / (1.0 + jnp.exp(-self.k * (ls - self.l0)))
        
        # Growth: c' = c * (1 + g * filter)
        # We keep l=0 (mean radius) constant roughly? 
        # Or just let it grow? The prompt says "amplifies high-frequency harmonics".
        # Typically global growth is separate.
        # Let's assume g applies strictly to the 'buckling' modes.
        
        amplification = 1.0 + g * filter_val
        return coeffs * amplification

    def get_radius_function(self, coeffs: jax.Array) -> Callable[[float, float], float]:
        """
        Returns a function R(theta, phi) evaluated from coefficients.
        """
        
        # We need a way to evaluate SH at (theta, phi)
        # e3nn_jax.spherical_harmonics takes vectors (x, y, z)
        
        def radius_fn(theta, phi):
            # Convert to cartesian unit vector
            x = jnp.sin(theta) * jnp.cos(phi)
            y = jnp.sin(theta) * jnp.sin(phi)
            z = jnp.cos(theta)
            vec = jnp.array([x, y, z])
            
            # Evaluate SH basis
            # irreps = e3nn.Irreps.spherical_harmonics(lmax)
            # But we need to cache this irreps definition ideally?
            # It's static, so it should be fine.
            # Construct Irreps string "0e + 1o + 2e ..."
            # parity (-1)^l: 0->1(e), 1->-1(o), 2->1(e)...
            irreps_str = " + ".join([f"{l}{'e' if (-1)**l == 1 else 'o'}" for l in range(self.lmax + 1)])
            irreps = e3nn.Irreps(irreps_str)
            
            # Note: e3nn spherical_harmonics normalization might differ. 
            # We assume 'component' normalization or 'integral' normalization
            # normalize=True ensures integral of Y^2 over sphere is 1 or 4pi?
            # e3nn default is component-wise unit variance? 
            # We will perform a standard check later.
            
            Y = e3nn.spherical_harmonics(irreps, vec, normalize=True, normalization='integral') 
            # normalization='integral' -> orthonormality over sphere? Check docs if available.
            # actually 'component' is default. 'integral' means integral |Y|^2 = 1.
            
            # coeffs dot basis
            return jnp.dot(coeffs, Y.array)
            
        return radius_fn

    def compute_shape_index_map(self, coeffs: jax.Array, theta_grid: jax.Array, phi_grid: jax.Array):
        """
        Computes the Shape Index at given grid points.
        """
        R_fn = self.get_radius_function(coeffs)
        
        # Define the position vector function x(theta, phi)
        def pos_fn(theta, phi):
            r = R_fn(theta, phi)
            return r * jnp.array([
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta)
            ])
        
        # First derivatives
        jac_fn = jax.jacfwd(pos_fn, argnums=(0, 1)) # returns (dx/dtheta, dx/dphi)
        
        # Second derivatives
        hess_fn = jax.hessian(pos_fn, argnums=(0, 1)) 
        # hessian returns ((d2x/dtheta2, d2x/dthedphi), (d2x/dphidtheta, d2x/dphi2))
        
        vmap_jac = jax.vmap(jac_fn)
        vmap_hess = jax.vmap(hess_fn)
        
        # Compute for all grid points
        # (N, 3)
        X_theta, X_phi = vmap_jac(theta_grid, phi_grid) 
        
        # (N, 2, 2, 3) 
        # we want X_tt, X_tp, X_pp
        H = vmap_hess(theta_grid, phi_grid)
        X_theta_theta = H[0][0]
        X_theta_phi = H[0][1] # or H[1][0]
        X_phi_phi = H[1][1]
        
        # Metric Tensor coefficients (First Fundamental Form)
        E = jnp.sum(X_theta * X_theta, axis=-1)
        F = jnp.sum(X_theta * X_phi, axis=-1)
        G = jnp.sum(X_phi * X_phi, axis=-1)
        
        # Surface Normal
        cross = jnp.cross(X_theta, X_phi)
        norm_cross = jnp.linalg.norm(cross, axis=-1)
        n = cross / (norm_cross[..., None] + 1e-9)
        
        # Second Fundamental Form coefficients
        L = jnp.sum(X_theta_theta * n, axis=-1)
        M = jnp.sum(X_theta_phi * n, axis=-1)
        N = jnp.sum(X_phi_phi * n, axis=-1)
        
        # Principal Curvatures
        # Gaussian Curvature K
        denom = E * G - F**2
        denom = jnp.where(denom < 1e-9, 1e-9, denom) # Avoid division by zero
        
        K_gauss = (L * N - M**2) / denom
        
        # Mean Curvature H
        # Note: With outward normal, L, N are negative for a sphere.
        # We negate H so that a convex sphere has positive curvature (SI ~ +1)
        H_mean = - (E * N + G * L - 2 * F * M) / (2 * denom)
        
        # Shape Index
        # S = 2/pi * arctan( (k1+k2) / (k2-k1) )
        # H = (k1+k2)/2
        # K = k1*k2
        # k1, k2 = H +/- sqrt(H^2 - K)
        # k1+k2 = 2H
        # k2-k1 = 2*sqrt(H^2 - K) (assuming k2 >= k1)
        # S = 2/pi * arctan( 2H / (2*sqrt(H^2 - K)) ) = 2/pi * arctan( H / sqrt(H^2 - K) )
        
        discriminant = H_mean**2 - K_gauss
        discriminant = jnp.maximum(discriminant, 0.0) # Ensure non-negative
        
        si_num = H_mean
        si_den = jnp.sqrt(discriminant)
        
        # Handle Umbical points (diff ~ 0)
        # If den is 0, arctan is +/- pi/2
        si = (2.0 / jnp.pi) * jnp.arctan2(si_num, si_den)
        
        return si, H_mean, K_gauss

    def __call__(self, coeffs_initial: jax.Array, g: float, key: jax.Array):
        """
        Runs the growth simulation and returns statistics.
        """
        # 1. Apply growth
        coeffs_grown = self.growth_operator(coeffs_initial, g)
        
        # 2. Sample points for SI calculation
        # Generate a Fibonacci lattice or similar
        # For differentiability, we use a fixed grid relative to lmax
        # 2*lmax usually good
        
        N_samples = 2000 # Enough for robust stats
        
        # Golden angle spiral
        indices = jnp.arange(0, N_samples, dtype=float) + 0.5
        phi = jnp.pi * (1 + 5**0.5) * indices
        costheta = 1 - 2 * indices / N_samples
        theta = jnp.arccos(costheta)
        
        si_map, h_map, k_map = self.compute_shape_index_map(coeffs_grown, theta, phi)
        
        # Filter NaN or invalid
        # valid = jnp.isfinite(si_map)
        
        return {
            'coeffs': coeffs_grown,
            'mean_si': jnp.mean(si_map),
            'std_si': jnp.std(si_map),
            'shape_index_distribution': si_map,
            'mean_curvature': h_map,
            'gaussian_curvature': k_map
        }
