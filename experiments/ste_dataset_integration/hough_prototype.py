import jax
import jax.numpy as jnp
import numpy as np
import time
from dmipy_jax.utils.spherical_harmonics import sh_basis_real_analytic, cart2sphere

def hough_benchmark():
    print("--- JAX Hough Tractography Prototype ---")
    
    # 1. Setup Synthetic Volume (SH Coefficients)
    # 64x64x64 volume, Lmax=4 -> 15 coeffs
    vol_shape = (64, 64, 64)
    lmax = 4
    n_coeffs = int((lmax + 1) * (lmax + 2) / 2)
    
    print(f"Generating synthetic SH volume {vol_shape} with {n_coeffs} coeffs...")
    # Random coeffs
    key = jax.random.PRNGKey(0)
    sh_volume = jax.random.normal(key, vol_shape + (n_coeffs,))
    
    # 2. Define Curve Model
    # Simple polynomial curve: r(t) = P(t)
    # We'll use 2nd order polynomial for X, Y, Z.
    # r(t) = c0 + c1*t + c2*t^2
    # t in [0, 1]
    # coeffs shape: (3, 3) -> (dims, order+1)
    
    def get_curve_points(coeffs, n_points=50):
        t = jnp.linspace(0, 1, n_points)
        # coeffs: (3, 3) -> [x_params, y_params, z_params]
        # x(t) = c[0,0] + c[0,1]*t + c[0,2]*t^2
        
        # Powers of t: (3, n_points)
        t_pow = jnp.stack([jnp.ones_like(t), t, t**2], axis=0)
        
        # positions: (3, 3) @ (3, n_points) -> (3, n_points)
        positions = coeffs @ t_pow
        return positions.T # (n_points, 3)

    def sample_sh_field(sh_vol, positions):
        # Interpolate SH coeffs at positions
        # sh_vol: (X, Y, Z, C)
        # positions: (N, 3) in grid coordinates
        
        # map_coordinates requires coordinates to be first axis: (ndim, n_points)
        coords = positions.T 
        
        # We need to map for EACH coefficient channel.
        # Or treat C as extra dim? map_coordinates allows batching? 
        # map_coordinates(input, coords) interpolates input.
        # If input has shape (X, Y, Z, C), and we map with (3, N), behavior depends on implementation.
        # Standard scipy map_coordinates handles arbitrary ranks but coords must match input rank.
        # We want to interpolate spatial dims (0,1,2) but keep channel dim (3) intact.
        # Trick: Move C to front (C, X, Y, Z), vmap over C.
        
        vol_T = jnp.moveaxis(sh_vol, -1, 0) # (C, X, Y, Z)
        
        def interp_channel(vol_chan):
             return jax.scipy.ndimage.map_coordinates(vol_chan, coords, order=1, mode='constant', cval=0.0)
        
        # vmap over channels
        samples = jax.vmap(interp_channel)(vol_T) # (C, N_points)
        return samples.T # (N_points, C)

    def score_curve(curve_coeffs, sh_vol):
        # 1. Generate points
        points = get_curve_points(curve_coeffs) # (N, 3)
        
        # 2. Compute Tangents (finite diff)
        # Simple diff
        tangents = jnp.gradient(points, axis=0) 
        # Normalize
        norms = jnp.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / (norms + 1e-9)
        
        # 3. Interpolate SH coeffs at points
        local_sh = sample_sh_field(sh_vol, points) # (N, C)
        
        # 4. Evaluate ODF along tangent
        # tangent (N, 3) -> (r, theta, phi)
        r, theta, phi = cart2sphere(tangents[:,0], tangents[:,1], tangents[:,2])
        
        # Basis at tangents
        basis = sh_basis_real_analytic(theta, phi, lmax) # (N, C)
        
        # 5. Dot product
        # ODF(u) = sum(c_i * Y_i(u))
        odf_values = jnp.sum(local_sh * basis, axis=1)
        
        # 6. Sum scores (integrate)
        total_score = jnp.sum(odf_values)
        return total_score

    # 3. Batching
    
    # Generate 100,000 random curves
    # Random 2nd order polys
    # Start inside volume (32,32,32) approx
    n_curves = 100_000
    print(f"Generating {n_curves} candidate curves...")
    
    k1, k2 = jax.random.split(key)
    # Random start points (32 +/- 10)
    start_pos = 32 + 10 * jax.random.normal(k1, (n_curves, 3, 1))
    # Random directions/curvature
    params = jax.random.normal(k2, (n_curves, 3, 2)) * 5.0 # c1, c2
    
    batch_coeffs = jnp.concatenate([start_pos, params], axis=2) # (N, 3, 3)
    
    # JIT the batch scorer
    print("Compiling global scorer...")
    
    # vmap score_curve over coeffs (axis 0 of batch_coeffs)
    # sh_vol is fixed (broadcasted)
    batched_score = jax.jit(jax.vmap(lambda c: score_curve(c, sh_volume)))
    
    # Trigger compile
    t0 = time.time()
    scores = batched_score(batch_coeffs[:10])
    scores.block_until_ready()
    print(f"Compilation finished in {time.time() - t0:.2f}s")
    
    # Run full batch
    print(f"Scoring {n_curves} curves...")
    t_start = time.time()
    scores = batched_score(batch_coeffs)
    scores.block_until_ready()
    duration = time.time() - t_start
    
    print(f"--- Results ---")
    print(f"Total time: {duration:.4f}s")
    print(f"Rate: {n_curves / duration:.2f} cubic_curves/sec")
    print(f"Score mean: {jnp.mean(scores)}")

if __name__ == "__main__":
    hough_benchmark()
