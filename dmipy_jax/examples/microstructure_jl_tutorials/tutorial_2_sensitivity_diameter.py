
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dmipy_jax.signal_models.cylinder_models import CallaghanRestrictedCylinder
from dmipy_jax.acquisition import JaxAcquisition

def main():
    print("Tutorial 2: Sensitivity Range of Axon Diameter Index")
    
    # 1. Define Acquisition
    # To probe diameter, we need strong gradients and suitable diffusion times.
    # Protocol: 
    #   Gradient strength G: 300 mT/m (Standard Connectome) or up to 300mT/m
    #   Delta/delta = 40/10 ms ?
    # Let's vary G to varying b-values.
    
    # Parameters for simulation
    im = JaxAcquisition(
        bvalues=jnp.array([1.0]), # dummy
        gradient_directions=jnp.array([[1.0, 0.0, 0.0]]) # dummy
    )
    # We will construct calls manually for plotting sensitivity curve 
    # Signal vs Diameter for a specific acquisition parameter set.
    
    # Protocol Parameters matching typical "Axon Diameter" sensitivity analysis:
    # G = 300 mT/m = 0.3 T/m
    # delta = 10 ms = 0.01 s
    # Delta = 30 ms = 0.03 s
    # effective diffusion time tau = Delta - delta/3
    # b-value approx: (gamma * G * delta)^2 * (Delta - delta/3)
    
    gamma = 2.675e8 # rad/s/T
    G_max = 0.3 # T/m (300 mT/m)
    delta = 0.01 # s
    Delta = 0.03 # s
    
    b_max = (gamma * G_max * delta)**2 * (Delta - delta/3.0)
    print(f"Max b-value: {b_max * 1e-6:.2f} s/mm^2")
    
    # We will check signal for 3 b-values/G-values:
    # 1. G = 100 mT/m
    # 2. G = 200 mT/m
    # 3. G = 300 mT/m
    
    gs = jnp.array([0.1, 0.2, 0.3]) # T/m
    bvals = (gamma * gs * delta)**2 * (Delta - delta/3.0) # s/m^2
    
    # Directions: Perpendicular to cylinder is where sensitivity exists.
    # Parallel signal is just exp(-b*D_par), decay depends on D_par, not diameter.
    # So we simulate gradient perpendicular to cylinder.
    # Cylinder along Z (0,0,1). Gradient along X (1,0,0).
    mu = jnp.array([0.0, 0.0]) # Z-axis (theta=0, phi=0) spherical
    
    # bvecs perpendicular
    gradient_directions = jnp.array([[1.0, 0.0, 0.0]] * len(gs))
    
    # 2. Model
    # Callaghan Restricted Cylinder
    cyl = CallaghanRestrictedCylinder(mu=mu, lambda_par=1.7e-9, diffusion_perpendicular=1.7e-9)
    
    # 3. Vary Diameter
    diameters = jnp.linspace(0.1e-6, 12e-6, 50) # 0.1um to 12um
    
    # Store results
    signals = []
    
    # We vmap over diameters
    def simulate_for_diameter(d):
        # Broadcast d to all b-values
        # But wait, cylinder model expects specific shapes or broadcasts?
        # cylinder_model call: bvals (N,), bvecs (N,3), diameter scalar.
        # We pass diameter.
        
        # We simulate for all 3 G-values at once
        S = cyl(bvals, gradient_directions, 
               diameter=d, big_delta=Delta, small_delta=delta)
        return S
        
    # vmap over diameters
    simulate_all = jax.vmap(simulate_for_diameter)
    
    signal_curves = simulate_all(diameters) # Shape (50, 3)
    
    # 4. Visualize
    plt.figure(figsize=(10, 6))
    
    for i in range(len(gs)):
        g_val = gs[i] * 1000 # mT/m
        plt.plot(diameters * 1e6, signal_curves[:, i], label=f'G = {g_val:.0f} mT/m')
        
    plt.xlabel('Axon Diameter (um)')
    plt.ylabel('Signal Attenuation (Perpendicular)')
    plt.title('Tutorial 2: Sensitivity of Singnal to Axon Diameter\n(Delta=30ms, delta=10ms)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=2.0, color='r', linestyle='--', alpha=0.3, label='2um limit')
    
    output_path = 'tutorial_2_output.png'
    plt.savefig(output_path)
    print(f"Sensitivity plot saved to {output_path}")

if __name__ == "__main__":
    main()
