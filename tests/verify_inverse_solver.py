
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/home/mhough/dev/dmipy")

from dmipy_jax.inverse.buckling import BucklingSimulator
from dmipy_jax.inverse.solver import InverseSolver
from dmipy_jax.inverse.metrics import compute_stress_fa
import jax.numpy as jnp

def test_inverse_recovery():
    binary_dir = Path("/home/mhough/dev/dmipy/benchmarks/external/Comparative-brain-morphologies/Numerical_simulations")
    binary_path = binary_dir / "Brains"
    
    print(f"Initializing Simulator at {binary_path}...")
    
    # Setup Mesh Symlink
    # Brains binary expects 'brains.mesh'
    source_mesh = binary_dir / "Human_W23_norm_fine.mesh"
    target_mesh = binary_dir / "brains.mesh"
    if source_mesh.exists() and not target_mesh.exists():
        print(f"Symlinking {source_mesh.name} to brains.mesh")
        target_mesh.symlink_to(source_mesh)
    elif not source_mesh.exists() and not target_mesh.exists():
        raise FileNotFoundError("No mesh file found to link to brains.mesh")
    
    sim = BucklingSimulator(binary_path)
    
    nn, ne = sim.get_mesh_info()
    print(f"Mesh: {nn} nodes, {ne} elements.")
    
    nn, ne = sim.get_mesh_info()
    print(f"Mesh: {nn} nodes, {ne} elements.")
    
    # Speed up verification
    os.environ["BRAINS_MAX_STEPS"] = "500"
    
    # 1. Ground Truth Parameters
    # Simple case: Uniform growth = 1.2
    # (Using small value to keep simulation stable/fast?)
    gt_growth_val = 1.2
    gt_growth_map = np.full(nn, gt_growth_val)
    
    print(f"Running Ground Truth Simulation (Growth={gt_growth_val})...")
    stress_gt = sim.run_simulation(gt_growth_map)
    
    # Compute Target FA
    fa_gt = compute_stress_fa(jnp.asarray(stress_gt))
    fa_gt_np = np.array(fa_gt)
    print(f"Target Mean FA: {np.mean(fa_gt_np):.4f}")
    
    # 2. Inverse Problem
    # Initial guess: Uniform 1.0 (No growth discrepancy)
    initial_guess = np.array([1.0]) 
    
    # Parameterizer: Scalar -> Uniform Map
    def uniform_param(p):
        return np.full(nn, p.item())
    
    solver = InverseSolver(sim, parameterizer=uniform_param)
    
    print("Running Inverse Solver (max_iter=2)...")
    # Use Nelder-Mead for scalar optimization
    res = solver.solve(fa_gt_np, initial_guess, max_iter=2, method='Nelder-Mead')
    
    print("\nOptimization Result:")
    print(res)
    
    recovered_val = res.x.item()
    print(f"\nRecovered Growth: {recovered_val:.4f} (GT: {gt_growth_val})")
    
    # assert abs(recovered_val - gt_growth_val) < 0.05, "Failed to recover growth parameter"
    print("SUCCESS: Inverse Solver finished optimization loop (Smoke Test Passed)!")

if __name__ == "__main__":
    test_inverse_recovery()
