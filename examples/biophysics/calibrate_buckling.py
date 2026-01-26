import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
from dmipy_jax.biophysics.buckling_layer import BucklingLayer

def calibrate():
    # 1. Initialize Layer
    lmax = 10
    layer = BucklingLayer(lmax=lmax, l0=2.0, k=5.0)
    
    # 2. Setup Initial Conditions (a sphere with small perturbations)
    # SH coefficients layout for e3nn (or just generic index)
    # We treated them as a flat vector in the layer
    num_coeffs = (lmax + 1)**2
    coeffs = jnp.zeros(num_coeffs)
    
    # l=0, m=0 (mean radius = 1.0)
    # Norm check: Y00 = 1/sqrt(4pi). Integral Y00^2 = 1.
    # If we want R=1 everywhere, coeffs[0] * Y00 = 1 => coeffs[0] = sqrt(4pi)
    # e3nn might use different normalization.
    # We will empirically check or use 1.0 if component normalized.
    # Let's assume standard 'integral' normalization
    coeffs = coeffs.at[0].set(jnp.sqrt(4 * jnp.pi)) 
    
    # Add random noise to l=3, 4, ... to allow buckling
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, (num_coeffs,)) * 0.01
    
    # Mask out l=0, 1 (keep position and size roughly stable)
    # l=0 is index 0
    # l=1 is indices 1, 2, 3
    mask = jnp.ones(num_coeffs)
    mask = mask.at[0].set(0.0)
    mask = mask.at[1:4].set(0.0)
    
    coeffs_initial = coeffs + noise * mask
    
    # 3. Sweep Growth Ratio g
    g_values = jnp.linspace(0.0, 10.0, 20)
    
    @jax.jit
    def eval_g(g):
        stats = layer(coeffs_initial, g, key)
        return stats
        
    results = []
    for g in g_values:
        stats = eval_g(g)
        results.append(stats)
        
    mean_sis = jnp.array([r['mean_si'] for r in results])
    std_sis = jnp.array([r['std_si'] for r in results])
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(g_values, mean_sis, '-o', label='Mean Shape Index')
    plt.fill_between(g_values, mean_sis - std_sis, mean_sis + std_sis, alpha=0.3)
    plt.xlabel('Growth Ratio (g)')
    plt.ylabel('Shape Index')
    plt.title('Buckling Layer Calibration')
    plt.grid(True)
    plt.legend()
    plt.savefig('buckling_calibration.png')
    print("Saved buckling_calibration.png")
    
    # Check if trend is increasing (or changing)
    print("Growth Values:", g_values)
    print("Mean Shape Index:", mean_sis)

if __name__ == '__main__':
    calibrate()
