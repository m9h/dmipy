
import jax
import jax.numpy as jnp
import numpy as np
import time
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.models.c_noddi import CNODDI
from dmipy_jax.fitting.optimization import OptimistixFitter
from dmipy_jax.core.pinns import BlochTorreyPINN, solve_bloch_torrey, physics_loss

def run_demo():
    print("================================================================")
    print("   DMIPY-JAX END-TO-END DEMO")
    print("================================================================")
    print("Hardware: ", jax.devices()[0])
    
    # =========================================================================
    # PART 1: INVERSE PROBLEM - MICROSTRUCTURE FITTING
    # =========================================================================
    print("\n----------------------------------------------------------------")
    print("PART 1: The Inverse Problem (Scanning & Reconstruction)")
    print("----------------------------------------------------------------")
    
    # 1.1 Generate Synthetic Data (LTE)
    print("--> Generatng Synthetic Data (C-NODDI Ground Truth)...")
    # 3 Shells: b=1000, 2000, 3000. 30 dirs each.
    # Total 90 measurements.
    bvals = []
    bvecs = []
    for b in [1000, 2000, 3000]:
        n_dirs = 30
        theta = np.linspace(0, np.pi, n_dirs)
        phi = np.linspace(0, 2*np.pi, n_dirs)
        
        gx = np.sin(theta) * np.cos(phi)
        gy = np.sin(theta) * np.sin(phi)
        gz = np.cos(theta)
        
        bvals.append(np.ones(n_dirs) * b * 1e6) # SI units s/m^2
        bvecs.append(np.stack([gx, gy, gz], axis=1))
        
    bvals = np.concatenate(bvals)
    bvecs = np.concatenate(bvecs)
    
    acq_lte = JaxAcquisition(bvals, bvecs)
    model = CNODDI()
    
    # True Params: [theta, phi, f_stick, f_iso]
    # Stick along Z [0, 0]
    true_params = jnp.array([0.0, 0.0, 0.6, 0.0])
    
    # Generate clean signal
    sig_lte = model(true_params, acq_lte)
    
    # Add Noise (SNR=50)
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, sig_lte.shape) * (1.0/50.0)
    data_lte = jnp.abs(sig_lte + noise) # Simple magnitude
    
    print(f"   Data Shape: {data_lte.shape}")
    print(f"   SNR: ~50")
    
    # 1.2 Fit using Optimistix
    print("\n--> Fitting Data (Optimistix + JAX)...")
    ranges = [
        (0.0, np.pi),   # theta
        (0.0, 2*np.pi), # phi
        (0.0, 1.0),     # f_stick
        (0.0, 1.0)      # f_iso
    ]
    scales = jnp.array([1.0, 1.0, 1.0, 1.0])
    
    fitter = OptimistixFitter(model, ranges, scales=scales)
    
    # Fit single voxel (jit compiled)
    t0 = time.time()
    fit_fn = jax.jit(fitter.fit)
    
    # Initial Guess
    init_params = jnp.array([1.5, 0.0, 0.5, 0.1])
    
    # Sigma is needed for Rician Loss
    sigma_est = 1.0 / 50.0
    
    fitted, res = fit_fn(data_lte, acq_lte, init_params, sigma_est)
    t1 = time.time()
    
    print(f"   Fit Time (Single Voxel + Compile): {t1-t0:.4f} s")
    print(f"   Ground Truth: {true_params}")
    print(f"   Fitted      : {fitted}")
    print(f"   Error (f_stick): {abs(fitted[2] - true_params[2]):.6f}")
    
    # 1.3 Multidimensional Diffusion (STE)
    print("\n--> Multidimensional Diffusion Check (STE Signal)")
    # Generate STE (Spherical Tensor) for same b-value (3000)
    b_ste = 3000 * 1e6
    # B = (b/3) * I
    B_ste = jnp.eye(3) * (b_ste / 3.0)
    # Make array of 1 measurement
    btensors_ste = jnp.stack([B_ste])
    
    acq_ste = JaxAcquisition(bvalues=jnp.array([b_ste]), gradient_directions=jnp.zeros((1,3)), btensors=btensors_ste)
    
    # Signal for STE
    # Should be exp(-b * trace(D)/3)
    # D_stick Trace = lambda_par (since lambda_perp=0)
    # S_stick = exp(-b * lambda_par / 3)
    
    sig_ste = model(true_params, acq_ste)
    expected_ste = jnp.exp(-b_ste * 1.7e-9 / 3.0)
    
    print(f"   STE Signal (Stick): {sig_ste[0]:.6f}")
    print(f"   Expected Isotropic: {expected_ste:.6f}")
    
    # Comparison with LTE at same b-value along stick
    # B_lte_par = b * diag([0,0,1]) -> Trace(B D) = b * lambda_par
    # S_lte = exp(-b * lambda_par)
    
    S_lte_predicted = jnp.exp(-b_ste * 1.7e-9)
    print(f"   LTE Signal (Parallel): {S_lte_predicted:.6f}")
    print("   Observation: STE preserves more signal than parallel LTE (due to b/3 scaling),")
    print("   but critically it effectively measures 'Mean Diffusivity' of the compartment")
    print("   without orientation dependence.")

    # =========================================================================
    # PART 2: FORWARD PROBLEM - SIMULATION (PINN)
    # =========================================================================
    print("\n----------------------------------------------------------------")
    print("PART 2: The Forward Problem (Physics Simulation with PINN)")
    print("----------------------------------------------------------------")
    
    print("--> Solving 1D Bloch-Torrey Equation (dM/dt - D*d2M/dx2 + ... = 0)")
    
    # Simulate a diffusion process with D=1.0 (normalized)
    D_sim = 0.5
    G_sim = 5.0 # High gradient
    
    print(f"   Physics: D={D_sim}, Gradient={G_sim}")
    
    key_pinn = jax.random.PRNGKey(101)
    
    # Collocation points
    # x in [-1, 1], t in [0, 0.5]
    n_pts = 200
    coords = jnp.stack([
        jnp.linspace(-1, 1, n_pts),
        jnp.linspace(0, 0.5, n_pts)
    ], axis=1)
    
    print("   Training PINN (100 steps)...")
    t0_pinn = time.time()
    
    final_pinn, sol = solve_bloch_torrey(
        key=key_pinn,
        coords=coords,
        D=D_sim,
        gamma=1.0, 
        G=G_sim,
        steps=100,
        lr=1e-3
    )
    t1_pinn = time.time()
    
    loss_final = physics_loss(final_pinn.model, coords, D_sim, 1.0, G_sim)
    
    print(f"   Training Time: {t1_pinn - t0_pinn:.4f} s")
    print(f"   Final Physics Residual: {loss_final:.6f}")
    
    print("\n================================================================")
    print("DEMO COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    run_demo()
