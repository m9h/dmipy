
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
from dmipy_jax.acquisition import JaxAcquisition

def main():
    print("Tutorial 1: Building a Microstructure Model in dmipy-jax")
    
    # 1. Define Acquisition
    # We'll use a simple Multi-Shell scheme: b=0, 1000, 2000, 3000
    # 20 directions per shell
    bvals_shell = jnp.array([1000.0, 2000.0, 3000.0]) * 1e6 # SI units s/m^2
    # Create simple directions (just random for demo or distributed)
    # For a tutorial, let's just make up some directions, or use a helper if available.
    # We'll just generate random ones for simplicity of this standalone script.
    key = jax.random.PRNGKey(42)
    n_dirs_per_shell = 20
    n_shells = 3
    
    # Random directions on sphere
    vecs = jax.random.normal(key, (n_shells * n_dirs_per_shell, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    # Full bvals/bvecs
    bvals = jnp.concatenate([jnp.zeros(5), jnp.repeat(bvals_shell, n_dirs_per_shell)])
    bvecs = jnp.concatenate([jnp.zeros((5, 3)), vecs])
    # Normalize bvecs (b0s can have 0 vector or unit vector, dmipy usually handles either)
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    print(f"Acquisition defined: {len(bvals)} measurements.")

    # 2. Define Components
    stick = C1Stick()
    zeppelin = G2Zeppelin()
    ball = G1Ball()
    
    # 3. Define the Composite Model Function
    # Model: (1 - f_iso) * (f_int * Stick + (1 - f_int) * Zeppelin) + f_iso * Ball
    # Parameters: 
    #   mu (orientation)
    #   lambda_par (parallel diffusivity)
    #   lambda_perp (perpendicular diffusivity)
    #   f_int (intra-axonal volume fraction)
    #   f_iso (isotropic volume fraction)
    #   D_iso (isotropic diffusivity - usually fixed to 3e-9)
    
    def my_tissue_model(params, acquisition):
        # Unpack parameters
        theta, phi = params[0], params[1]
        mu = jnp.array([theta, phi])
        
        lambda_par = params[2] # Parallel Diffusivity
        lambda_perp = params[3] # Perpendicular Diffusivity
        
        f_int = params[4] # Stick fraction (relative to tissue)
        f_iso = params[5] # CSF fraction
        
        D_iso = 3.0e-9 # Fixed free water diffusivity
        
        # calculate tissue fraction
        f_tissue = 1.0 - f_iso
        
        # Calculate Signals
        S_stick = stick(acquisition.bvalues, acquisition.gradient_directions, 
                        mu=mu, lambda_par=lambda_par)
        
        S_zeppelin = zeppelin(acquisition.bvalues, acquisition.gradient_directions,
                              mu=mu, lambda_par=lambda_par, lambda_perp=lambda_perp)
                              
        S_ball = ball(acquisition.bvalues, lambda_iso=D_iso)
        
        # Combine
        S_tissue = f_int * S_stick + (1.0 - f_int) * S_zeppelin
        S_total = f_tissue * S_tissue + f_iso * S_ball
        
        return S_total

    # 4. Simulate a signal
    # Ground Truth Parameters
    # Orientation: z-axis (theta=0, phi=0)
    gt_theta, gt_phi = 0.0, 0.0
    gt_lambda_par = 1.7e-9
    gt_lambda_perp = 0.2e-9
    gt_f_int = 0.6
    gt_f_iso = 0.1 # 10% CSF
    
    gt_params = jnp.array([gt_theta, gt_phi, gt_lambda_par, gt_lambda_perp, gt_f_int, gt_f_iso])
    
    print("Simulating signal with GT params:", gt_params)
    
    S_sim = my_tissue_model(gt_params, acq)
    
    print("Simulation complete. Signal shape:", S_sim.shape)
    print("First 10 signal values:", S_sim[:10])
    
    # 5. Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(acq.bvalues, S_sim, 'o', alpha=0.6, label='Simulated Signal')
    plt.xlabel('b-value (s/m^2)')
    plt.ylabel('Signal Attenuation')
    plt.title('Tutorial 1: Simulated Signal (Stick + Zeppelin + Ball)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    output_path = 'tutorial_1_output.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
