import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
from dmipy_jax.io.mesh import create_sphere_mesh
from dmipy_jax.simulation.mesh_sim import Mesh, MatrixFormalismSimulator
from dmipy_jax.signal_models.sphere_models import SphereCallaghan, SphereGPD


def run_verification():
    print("Running Matrix Formalism Verification...")
    
    # 1. Physics Parameters
    R_target = 5e-6 # 5 microns
    D_0 = 2e-9      # 2 um^2/ms -> 2e-9 m^2/s
    
    # 2. Protocol
    # Big gradient to see restriction
    G_amp = 0.3 # 300 mT/m
    delta = 10e-3
    Delta = 20e-3
    
    # 3. Analytical Ground Truth
    # dmipy-jax Sphere model
    # It usually takes (acquisition, parameters).
    # Construct a minimal acquisition
    bvecs = jnp.array([[1.0, 0.0, 0.0]])
    bvals = jnp.array([1.0]) # Dummy, we'll manually compute q if needed, or rely on model internals
    # Actually Sphere model takes 'acquisition' object.
    
    # Instantiate Analytical Model
    # SphereGPD is best match for Restricted Diffusion (Murday-Cotts)
    # Note: SphereCallaghan is more general but JAX port has issues with Bessel deriv orders.
    sphere_ana = SphereGPD(diameter=R_target*2, diffusion_constant=D_0)
    
    # 4. Mesh Simulator
    # Create Unit Sphere
    verts, faces = create_sphere_mesh(radius=1.0, level=2) # icosahedron for now
    # Scale vertices to R_target
    mesh_phys = Mesh(vertices=verts * R_target, faces=faces)
    
    print(f"Mesh has {verts.shape[0]} vertices and {faces.shape[0]} faces.")
    sys.stdout.flush()

    # Instantiate Simulator
    # K=25 modes (Closure of L=4 shell: 1+3+5+7+9 = 25) to avoid degeneracy cutoff NaN
    sim = MatrixFormalismSimulator(mesh_phys, diffusivity=D_0, K=25)
    
    # Run Simulation
    signal_fem = sim(G_amp, delta, Delta)
    print(f"FEM Signal: {signal_fem:.6f}")
    
    # Analytical Signal
    # SphereGPD requires big_delta, small_delta in kwargs
    # Can also pass bvals/gradient_directions
    
    # Calculate b-value for this pulse
    gamma = 267.513e6
    q_val = gamma * G_amp * delta / (2 * jnp.pi) # m^-1
    # Stejskal Tanner b = (2pi q)^2 (Delta - delta/3)
    tau = Delta - delta / 3.0
    b_val = (2 * jnp.pi * q_val) ** 2 * tau
    
    # Pass via kwargs as required by __call__
    # Note: simple call expecting bvals/bvecs usually
    # But we can pass manual kwargs for GPD
    signal_ana_arr = sphere_ana(bvals=jnp.array([b_val]), gradient_directions=jnp.array([[1.0, 0.0, 0.0]]), big_delta=jnp.array([Delta]), small_delta=jnp.array([delta]))
    signal_ana = float(signal_ana_arr[0])
    print(f"Analytical Signal (GPD): {signal_ana:.6f}")
    
    err = abs(signal_fem - signal_ana) / signal_ana
    print(f"Relative Error: {err*100:.2f}%")
    sys.stdout.flush()

    
    # 5. Differentiability Check
    def simulate_at_radius(r_val):
        # Reinstate mesh and simulator
        # Note: Eigendecomposition is differentiable!
        m = Mesh(vertices=verts * r_val, faces=faces)
        s = MatrixFormalismSimulator(m, diffusivity=D_0, K=25)
        return s(G_amp, delta, Delta)
        
    print("Computing Gradient dSignal/dRadius...")
    val, grad_r = jax.value_and_grad(simulate_at_radius)(R_target)
    print(f"Signal at R={R_target*1e6:.1f}um: {val:.6f}")
    print(f"Grad dS/dR: {grad_r:.2e}")
    
    # Check physical intuition: Larger R -> Less restriction "locally" -> Wait.
    # For fixed Delta/delta/G?
    # In restricted diffusion (sphere), Signal ~ exp(-q^2 D_eff).
    # D_eff increases with R (up to D_0).
    # So Signal should DECREASE as R increases (more attenuation).
    # So Gradient should be NEGATIVE.
    
    if grad_r < 0:
        print("SUCCESS: Gradient is negative (Physical intuition holds).")
    else:
        print("WARNING: Gradient is non-negative (Check logic).")
        
    # 6. Multi-Gradient Curve
    G_vals = jnp.linspace(0.01, 0.4, 10)
    sigs = []
    for g in G_vals:
        sigs.append(sim(g, delta, Delta))
    sigs = jnp.array(sigs)
    
    print("Signal Curve vs G:", sigs)
    
if __name__ == "__main__":
    run_verification()
