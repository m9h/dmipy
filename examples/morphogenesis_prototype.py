"""
Morphogenesis Prototype: From Microstructure to Mechanical Stress.
Integrates sbi4dwi microstructural parameters with Kuhl-inspired CANNs.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from dmipy_jax.morphogenesis.cann import MorphoCANN
from dmipy_jax.signal_models.sandi import get_sandi_model
from dmipy_jax.acquisition import JaxAcquisition

def run_prototype():
    # 1. Setup Synthetic Acquisition (from sbi4dwi patterns)
    # Using a subset of HCP-like b-values
    bvalues = jnp.array([0, 1000, 2000, 3000] * 10) * 1e6 # s/m^2
    gradient_directions = jr.normal(jr.PRNGKey(0), (40, 3))
    gradient_directions /= jnp.linalg.norm(gradient_directions, axis=-1, keepdims=True)
    
    # Placeholder delta/Delta for SANDI (sphere models)
    delta = 0.02 # s
    Delta = 0.04 # s
    
    acq = JaxAcquisition(
        bvalues=bvalues,
        gradient_directions=gradient_directions,
        delta=delta,
        Delta=Delta
    )
    
    # 2. Generate Synthetic Microstructure (the "Estimated" parameters)
    # Voxel 1: High axonal density (White Matter proxy)
    # Voxel 2: High soma density (Gray Matter proxy)
    # [theta, phi, f_stick, f_sphere, f_ball, diameter, lambda_perp]
    params_wm = jnp.array([0.0, 0.0, 0.7, 0.05, 0.05, 10e-6, 0.2e-9])
    params_gm = jnp.array([0.0, 0.0, 0.2, 0.50, 0.10, 15e-6, 0.8e-9])
    
    # 3. Initialize MorphoCANN (The mechanical model)
    key = jr.PRNGKey(42)
    morpho_model = MorphoCANN(n_basis=4, key=key)
    
    # 4. Map Microstructure to Mechanical Stress
    # Let's assume a 10% tangential stretch (Buckling precursor)
    # F = [1.1, 0, 0; 0, 1.1, 0; 0, 0, 0.826] (Isochoric stretch)
    F = jnp.diag(jnp.array([1.1, 1.1, 1.0 / (1.1**2)]))
    
    # Fiber direction from mu (theta, phi)
    def get_a0(theta, phi):
        return jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])
    
    a0_wm = get_a0(params_wm[0], params_wm[1])
    a0_gm = get_a0(params_gm[0], params_gm[1])
    
    # Compute Cauchy Stress
    sigma_wm = morpho_model.cauchy_stress(F, a0_wm)
    sigma_gm = morpho_model.cauchy_stress(F, a0_gm)
    
    print("--- Morphogenesis Prototype Result ---")
    print(f"White Matter Proxy (f_ic=0.7) Stress:\n{sigma_wm[0,0]:.4f} kPa (approx)")
    print(f"Gray Matter Proxy (f_sphere=0.5) Stress:\n{sigma_gm[0,0]:.4f} kPa (approx)")
    
    # 5. Analysis
    # In a real growth model, if sigma_gm > sigma_wm (scaled by growth rate), 
    # we expect buckling to occur in the GM layer.
    
    diff = sigma_gm[0,0] - sigma_wm[0,0]
    print(f"\nStress Difference (GM - WM): {diff:.4f}")
    if diff > 0:
        print("Conclusion: Gray matter layer exhibits higher compressive stress, driving buckling.")
    else:
        print("Conclusion: White matter substrate is stiffer, resisting folding.")

if __name__ == "__main__":
    run_prototype()
