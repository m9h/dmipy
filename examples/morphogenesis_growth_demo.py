"""
Morphogenesis Growth Demo: Simulating Cortical Buckling.
Uses HyperelasticGrowthSolver to simulate a growing bilayer.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from dmipy_jax.morphogenesis.cann import MorphoCANN
from dmipy_jax.morphogenesis.growth_solver import HyperelasticGrowthSolver

def run_growth_demo():
    # 1. Setup Grid (Small 16x16x16 for speed)
    grid_shape = (16, 16, 16)
    dx = 1.0
    
    # 2. Initialize CANN (Mechanical Properties)
    key = jr.PRNGKey(42)
    cann = MorphoCANN(n_basis=4, key=key)
    
    # 3. Initialize Solver
    solver = HyperelasticGrowthSolver(cann, grid_shape, dx)
    
    # 4. Define Growth Field F_g
    # Bilayer: Growth in top half (z > 8)
    # Tangential growth (in X and Y directions)
    growth_ratio = 1.2
    F_g_iso = jnp.eye(3)[:, :, None, None, None]
    F_g_grown = jnp.diag(jnp.array([growth_ratio, growth_ratio, 1.0]))[:, :, None, None, None]
    
    # Create mask for top half
    Z_indices = jnp.arange(grid_shape[2])
    is_gm = Z_indices > 8 # (16,)
    
    # Broadcast to (3, 3, 16, 16, 16)
    # Mask: (16, 16, 16)
    mask_gm = jnp.broadcast_to(is_gm[None, None, :], grid_shape)
    
    # Compose F_g
    F_g = jnp.where(mask_gm[None, None, ...], F_g_grown, F_g_iso)
    
    # 5. Define Fiber Orientation a0 (Radial)
    # Fibers pointing along Z
    a0 = jnp.zeros((3,) + grid_shape)
    a0 = a0.at[2].set(1.0)
    
    # 6. Run Equilibrium Solver
    print("Starting growth simulation...")
    # Using small n_iters for demo
    u_final, history = solver.solve_equilibrium(F_g, a0, n_iters=50, lr=1e-2)
    
    # 7. Results Analysis
    print(f"Initial Energy: {history[0]}")
    print(f"Final Energy: {history[-1]}")
    
    # Check max displacement
    max_u = jnp.max(jnp.abs(u_final))
    print(f"Max Displacement: {max_u:.4f}")
    
    if history[-1] < history[0] and max_u > 0:
        print("Success: System reached lower energy state via deformation.")
    else:
        print("Warning: Energy did not decrease significantly or no deformation occurred.")

if __name__ == "__main__":
    run_growth_demo()
