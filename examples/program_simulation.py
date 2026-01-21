
"""
Testing Program Part 3: Simulation Validation (Diffrax Showcase).

Validates that the 'RestrictedCylinder' analytical formula matches
the 'DifferentiableWalker' SDE simulation for Connectome acquisition.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.simulation.differentiable_walker import DifferentiableWalker
from dmipy_jax.simulation.simulator import EndToEndSimulator
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.acquisition import JaxAcquisition

def run_simulation_program():
    print("=== Testing Program: Diffrax Simulation Validation ===")
    
    # 1. Setup Scenario
    # Single cylinder, radius=3um
    radius = 3.0e-6
    D0 = 1.7e-9
    
    # 2. Analytical Signal (RestrictedCylinder)
    print("Computing Analytical Signal (Soderman/Callaghan approx)...")
    cyl_model = RestrictedCylinder(diameter=radius*2, lambda_par=D0)
    
    # Scheme: High G (300 mT/m), Delta=30ms, delta=10ms
    # b ~ gamma^2 G^2 delta^2 (Delta - delta/3)
    # = (2.67e8 * 0.3 * 0.01)^2 * (0.03 - 0.0033)
    # = (8e5)^2 * 0.0267 ~ 6.4e11 * 0.0267 ~ 1.7e10 = 17,000 s/mm^2
    
    G_val = 0.3 # T/m
    delta = 0.01
    Delta = 0.03
    bval = (267.513e6 * G_val * delta)**2 * (Delta - delta/3.0)
    
    scheme = JaxAcquisition(
        bvalues=jnp.array([bval]),
        gradient_directions=jnp.array([[1.0, 0.0, 0.0]]),
        delta=jnp.array([delta]),
        Delta=jnp.array([Delta])
    )
    
    # Unpack scheme for direct model call
    bvals = scheme.bvalues
    bvecs = scheme.gradient_directions
    big_delta = scheme.Delta
    small_delta = scheme.delta
    
    # Pass timing as kwargs since we aliased them or updated cylinder_models
    kwargs = {'big_delta': big_delta, 'small_delta': small_delta}
    
    S_analytical = cyl_model(bvals, bvecs, mu=jnp.array([1.0, 0.0, 0.0]), **kwargs) 
    
    S_parallel = cyl_model(bvals, bvecs, mu=jnp.array([1.0, 0.0, 0.0]), **kwargs)
    S_perp = cyl_model(bvals, bvecs, mu=jnp.array([0.0, 1.0, 0.0]), **kwargs)
    
    print(f"Analytical Perp Signal: {S_perp[0]:.4f}")
    
    # 3. SDE Simulation (Diffrax)
    print("Computing SDE Simulated Signal (DifferentiableWalker)...")
    walker = DifferentiableWalker(
        diffusivity=D0, 
        radius=radius, 
        geometry_type='cylinder' # 2D restriction
    )
    sim = EndToEndSimulator(walker)
    
    key = jax.random.PRNGKey(0)
    # Simulate Perpendicular (Gradient=[1,0,0], Cylinder Axis=[0,0,1] implied Z-axis usually?)
    # Walker usually defines restriction in XY plane for 'cylinder'.
    # Gradient X -> Restricted.
    
    S_sim_perp = sim(
        G_amp=G_val, delta=delta, Delta=Delta, 
        key=key, N_particles=2000, dt=5e-5
    )
    
    print(f"Simulated Perp Signal: {S_sim_perp:.4f}")
    
    # Compare
    diff = jnp.abs(S_perp[0] - S_sim_perp)
    print(f"Difference: {diff:.4f}")
    
    if diff < 0.05:
        print("SUCCESS: Simulation matches Analytical within tolerance.")
    else:
        print("WARNING: Mismatch > 0.05. Check definitions (Radius vs Diameter, SDE bounds).")

if __name__ == "__main__":
    run_simulation_program()
