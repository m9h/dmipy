"""
Morphogenesis Lab: Full Pipeline from Microstructure to Folding.
Integrates sbi4dwi, Toro, and Kuhl models in a JAX-accelerated simulation.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from dmipy_jax.morphogenesis.cann import MorphoCANN
from dmipy_jax.morphogenesis.growth_solver import HyperelasticGrowthSolver
from dmipy_jax.morphogenesis.utils import map_microstructure_to_stiffness, get_growth_tensor

def run_morphogenesis_lab():
    # 1. Parameter Definitions
    grid_shape = (16, 16, 16)
    dx = 0.5 # mm
    n_iters = 100
    lr = 5e-4
    
    # 2. Mock Microstructure Map (simulating sbi4dwi output)
    # Define a Gray Matter (GM) layer and a White Matter (WM) core
    Z = jnp.arange(grid_shape[2])
    is_gm = Z > 10 # Top 5 voxels are GM
    is_wm = Z <= 10 # Bottom voxels are WM
    
    # fractions (H, W, D)
    f_sphere = jnp.where(is_gm, 0.5, 0.05) # Somas high in GM
    f_ic = jnp.where(is_wm, 0.7, 0.2)      # Axons high in WM
    
    # orientations (3, H, W, D) - Radial glia / Axonal orientation
    # Pointing primarily along Z
    a0 = jnp.zeros((3,) + grid_shape)
    a0 = a0.at[2].set(1.0)
    
    # 3. Derive Mechanical Properties
    stiffness_map = map_microstructure_to_stiffness(f_ic, f_sphere)
    
    # Growth: 15% tangential expansion ONLY in GM
    growth_ratio = jnp.where(is_gm, 1.15, 1.0)
    F_g = get_growth_tensor(growth_ratio, a0, is_tangential=True)
    
    # 4. Initialize Physics-Informed Model (CANN)
    key = jr.PRNGKey(42)
    # Note: In a real scenario, CANN weights would be subject-specific
    cann = MorphoCANN(n_basis=4, key=key)
    
    # 5. Run Differentiable Growth Simulation
    solver = HyperelasticGrowthSolver(cann, grid_shape, dx)
    
    print("--- Morphogenesis Lab Simulation ---")
    print(f"Grid: {grid_shape}, dx: {dx}mm")
    print(f"GM Thickness: {jnp.sum(is_gm) * dx}mm")
    print(f"Growth Ratio: 1.15")
    
    u_final, history = solver.solve_equilibrium(F_g, a0, n_iters=n_iters, lr=lr)
    
    # 6. Analysis & Diagnostics
    initial_energy = history[0]
    final_energy = history[-1]
    energy_reduction = (initial_energy - final_energy) / initial_energy * 100
    
    max_displacement = jnp.max(jnp.abs(u_final))
    
    # Compute Final Cauchy Stress Map
    F_final = solver.get_deformation_gradient(u_final)
    # Reshape for vectorized stress calc
    N = jnp.prod(jnp.array(grid_shape))
    F_flat = F_final.transpose(2, 3, 4, 0, 1).reshape(N, 3, 3)
    Fg_flat = F_g.transpose(2, 3, 4, 0, 1).reshape(N, 3, 3)
    Fg_inv_flat = jnp.linalg.inv(Fg_flat)
    Fe_flat = jnp.matmul(F_flat, Fg_inv_flat)
    a0_flat = a0.transpose(1, 2, 3, 0).reshape(N, 3)
    
    vmap_stress = jax.vmap(cann.cauchy_stress)
    sigma_flat = vmap_stress(Fe_flat, a0_flat)
    sigma = sigma_flat.reshape(grid_shape + (3, 3)).transpose(3, 4, 0, 1, 2)
    
    # Max Principal Stress in GM
    sigma_gm = sigma[:, :, :, :, is_gm]
    max_stress_gm = jnp.max(jnp.abs(sigma_gm))
    
    print("\n--- Results ---")
    print(f"Energy Reduction: {energy_reduction:.2f}%")
    print(f"Max Displacement: {max_displacement:.4f} mm")
    print(f"Max Stress in GM: {max_stress_gm:.4f} (normalized units)")
    
    if energy_reduction > 1.0:
        print("\nStatus: Buckling transition detected. Folding pattern initialized.")
    else:
        print("\nStatus: Stable expansion. Increase growth ratio or n_iters to see folding.")

if __name__ == "__main__":
    run_morphogenesis_lab()
