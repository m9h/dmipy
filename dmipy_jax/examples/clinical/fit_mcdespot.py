
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from dmipy_jax.models.mcdespot import McDESPOT, McDESPOTParameters

def run_fitting_demo():
    print("--> Starting McDESPOT Fitting Demo...")
    
    # 1. Ground Truth Parameters
    # Simulating a Single Voxel of White Matter
    true_params = McDESPOTParameters(
        f_myelin=0.15,      # 15% Myelin Water
        T1_myelin=450.0,    # ms
        T2_myelin=15.0,     # ms
        T1_ie=1000.0,       # ms
        T2_ie=80.0,         # ms
        off_resonance=0.0
    )
    
    print(f"Ground Truth: MWF={true_params.f_myelin:.2f}, "
          f"T1_m={true_params.T1_myelin}, T2_m={true_params.T2_myelin}")

    # 2. Define Protocol
    # SPGR VFA: Flip Angles 2 to 20 deg
    spgr_alphas = jnp.deg2rad(jnp.linspace(2, 20, 8))
    spgr_TR = 5.0 # ms
    
    # bSSFP VFA: Flip Angles 2 to 60 deg
    # Two Phase Cycles: 0 and 180 (band shifting)
    ssfp_alphas = jnp.deg2rad(jnp.linspace(2, 60, 8))
    ssfp_TR = 5.0 # ms
    
    model = McDESPOT()
    
    # 3. Simulate Data (The "Measurement")
    @jax.jit
    def forward_simulate(p: McDESPOTParameters):
        # SPGR data
        s_spgr = jax.vmap(lambda a: model(p, 'SPGR', spgr_TR, a))(spgr_alphas)
        
        # SSFP data (PC=0)
        s_ssfp_0 = jax.vmap(lambda a: model(p, 'SSFP', ssfp_TR, a, 0.0))(ssfp_alphas)
        
        # SSFP data (PC=180)
        s_ssfp_180 = jax.vmap(lambda a: model(p, 'SSFP', ssfp_TR, a, jnp.pi))(ssfp_alphas)
        
        return jnp.concatenate([s_spgr, s_ssfp_0, s_ssfp_180])

    data_true = forward_simulate(true_params)
    
    # Add Noise?
    # For robust demo, let's start noiseless. Then add noise if needed.
    
    # 4. Define Loss Function
    # We optimize a vector of 5 params: [f_m, T1_m, T2_m, T1_ie, T2_ie]
    # We fix Off-Resonance to 0 for this demo (or fit it too?)
    # Let's fit the 5 main params using Log-Space/Sigmoid scaling to enforce constraints.
    
    def unpack(theta):
        # theta is unconstrained
        # f_m: Sigmoid -> (0, 0.5)
        f_m = jax.nn.sigmoid(theta[0]) * 0.5 
        
        # T1_m: Softplus + shift -> (100, 800)
        T1_m = jax.nn.softplus(theta[1]) * 500.0 + 100.0
        # T2_m: Softplus -> (5, 40)
        T2_m = jax.nn.softplus(theta[2]) * 30.0 + 5.0
        
        # T1_ie: Softplus -> (500, 2000)
        T1_ie = jax.nn.softplus(theta[3]) * 1000.0 + 500.0
        # T2_ie: Softplus -> (40, 200)
        T2_ie = jax.nn.softplus(theta[4]) * 150.0 + 40.0
        
        return McDESPOTParameters(f_m, T1_m, T2_m, T1_ie, T2_ie, 0.0)

    @jax.jit
    def loss_fn(theta):
        p_est = unpack(theta)
        data_est = forward_simulate(p_est)
        # MSE
        return jnp.mean((data_est - data_true)**2)

    # 5. Optimization Loop (Adam)
    # Improved Initialization:
    # Initialize near "standard" WM values to avoid local minima
    # Standard WM: T1_m=450, T2_m=20, T1_ie=1000, T2_ie=80
    # Sigmoid(0) = 0.5 -> MWF * 0.5 = 0.25. (Standard is 0.1-0.2)
    # Log-Space init:
    # T1_m ~ 500 -> softplus_inv( (500-100)/500 ) approx 0.5
    
    init_theta = jnp.array([-1.0, 0.5, 0.0, 0.5, 0.5]) 
    # theta[0] = -1 -> sigmoid(-1)*0.5 = 0.26 * 0.5 = 0.13 (Good start for MWF)
    
    # Use Learning Rate Schedule
    scheduler = optax.cosine_decay_schedule(init_value=0.05, decay_steps=500, alpha=0.01)
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(init_theta)
    
    params = init_theta
    
    print("--> Fitting (Improved Init)...")
    for i in range(500):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if i % 50 == 0:
            p_curr = unpack(params)
            print(f"Iter {i}: Loss {loss:.6e} | MWF {p_curr.f_myelin:.3f}")


    # 6. Report Results
    final_p = unpack(params)
    print("\n--- Final Results ---")
    print(f"Recovered: MWF={final_p.f_myelin:.4f} (True: {true_params.f_myelin})")
    print(f"           T1_m={final_p.T1_myelin:.1f} (True: {true_params.T1_myelin})")
    print(f"           T2_m={final_p.T2_myelin:.1f} (True: {true_params.T2_myelin})")
    
    # 7. SCICO Recommendation
    print("\n[NOTE]: This voxel-wise fit is sensitive to initialization and noise.")
    print("For clinical reconstruction, we recommend using SCICO with Total Variation")
    print("regularization on the 'f_myelin' map to suppress spatial noise.")
    print(">> See fitting_plan.md for details.")

if __name__ == "__main__":
    run_fitting_demo()
