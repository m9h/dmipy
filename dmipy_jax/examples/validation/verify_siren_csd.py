
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.core.pinns import SIREN_CSD
from dmipy_jax.core.integration_grids import get_spherical_fibonacci_grid
from dmipy_jax.utils.spherical_harmonics import cart2sphere

def generate_synthetic_data(bvecs, bval, response_evals, fiber_dirs, weights, snr=None):
    """
    Generates synthetic signal for multi-fiber configuration.
    """
    n_meas = bvecs.shape[0]
    signal = jnp.zeros(n_meas)
    
    # Response parameters
    lam1 = response_evals[0]
    lam2 = response_evals[1]
    
    for i, fiber_dir in enumerate(fiber_dirs):
        # Cosine between fiber and bvecs
        cos_theta = jnp.dot(bvecs, fiber_dir)
        cos_sq = cos_theta**2
        sin_sq = 1.0 - cos_sq
        
        exponent = -bval * (lam1 * cos_sq + lam2 * sin_sq)
        fiber_signal = jnp.exp(exponent)
        
        signal += weights[i] * fiber_signal
        
    if snr is not None:
        key = jr.PRNGKey(0)
        sigma = 1.0 / snr
        noise = sigma * jr.normal(key, (n_meas, 2)) # Complex noise
        noisy_signal_complex = signal + noise[:, 0] + 1j * noise[:, 1]
        signal = jnp.abs(noisy_signal_complex)
        
    return signal

def main():
    print("Verifying SIREN CSD Implementation...")
    
    # 1. Setup Simulation
    bval = 3000.0 # High b-value for good angular resolution
    response_evals = jnp.array([1.7e-3, 0.2e-3, 0.2e-3])
    
    # Create b-vecs on sphere
    bvecs, _ = get_spherical_fibonacci_grid(n_points=64)
    
    # Define Ground Truth Fibers (Crossing at 90 deg)
    fiber1 = jnp.array([1.0, 0.0, 0.0])
    fiber2 = jnp.array([0.0, 1.0, 0.0])
    fiber_dirs = jnp.stack([fiber1, fiber2])
    weights = jnp.array([0.5, 0.5])
    
    print(f"Generating data for 2 fibers: {fiber1}, {fiber2}")
    data = generate_synthetic_data(bvecs, bval, response_evals, fiber_dirs, weights, snr=30)
    
    # 2. Initialize Model
    key = jr.PRNGKey(42)
    siren_csd = SIREN_CSD(
        response_evals=response_evals, 
        key=key, 
        sigma=1.0/30.0,
        n_integration_points=1000,
        hidden_features=128
    )
    
    # 3. Fit
    print("Fitting SIREN CSD...")
    # Use eqx.filter_jit regarding the model as an argument
    @eqx.filter_jit
    def fit_step(model, d, bvecs, b):
        return model.fit_voxel(d, bvecs, b)
    
    fitted_model, result = fit_step(siren_csd, data, bvecs, bval)
    print(f"Optimization Success: {result}")
    
    # 4. Evaluate FOD
    # Evaluate on a dense grid to find peaks
    eval_grid, _ = get_spherical_fibonacci_grid(n_points=2000)
    fod = fitted_model.get_fod(eval_grid)
    
    # 5. Extract Peaks (Simple argmax check for now, or threshold)
    # Just check if high FOD values align with fiber directions
    
    # Check alignment with Fiber 1
    dot1 = jnp.abs(jnp.dot(eval_grid, fiber1))
    mask1 = dot1 > 0.95
    max_fod1 = jnp.max(fod * mask1)
    
    # Check alignment with Fiber 2
    dot2 = jnp.abs(jnp.dot(eval_grid, fiber2))
    mask2 = dot2 > 0.95
    max_fod2 = jnp.max(fod * mask2)
    
    # Check background (orthogonal)
    ortho_dir = jnp.array([0.0, 0.0, 1.0])
    dot3 = jnp.abs(jnp.dot(eval_grid, ortho_dir))
    mask3 = dot3 > 0.95
    max_fod_ortho = jnp.max(fod * mask3)
    
    print(f"Peak FOD at Fiber 1 direction: {max_fod1:.4f}")
    print(f"Peak FOD at Fiber 2 direction: {max_fod2:.4f}")
    print(f"Background FOD (Orthogonal): {max_fod_ortho:.4f}")
    
    # Criterion: Peaks should be significantly higher than background
    factor = 2.0
    success = (max_fod1 > factor * max_fod_ortho) and (max_fod2 > factor * max_fod_ortho)
    
    if success:
        print("SUCCESS: Peaks detected at fiber directions significantly higher than background.")
    else:
        print("FAILURE: Peaks not clearly distinguished or too much background.")
        
if __name__ == "__main__":
    main()
