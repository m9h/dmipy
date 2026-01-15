import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.design.oed import optimize_protocol, d_optimality_loss
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick

def main():
    print("Setting up OED verification...")
    
    # 1. Setup Model
    # Target tissue parameters
    # lambda_par = 1.7e-9 m^2/s (typical in vivo)
    # mu = [pi/2, 0] (x-axis)
    target_params = {
        'lambda_par': 1.7e-9,
        'mu': jnp.array([jnp.pi/2, 0.0])
    }
    
    # Instantiate model
    model = C1Stick()
    
    # 2. Setup Initial Protocol
    # Start with suboptimal b-values (e.g., very low) and random directions
    N_meas = 10
    b_initial = jnp.ones(N_meas) * 1e6 # 1 s/mm^2 (very low)
    # Random directions
    key = jax.random.PRNGKey(0)
    vecs_initial = jax.random.normal(key, (N_meas, 3))
    vecs_initial = vecs_initial / jnp.linalg.norm(vecs_initial, axis=1, keepdims=True)
    
    delta = 0.010
    Delta = 0.020
    
    initial_acq = JaxAcquisition(
        bvalues=b_initial,
        gradient_directions=vecs_initial,
        delta=delta,
        Delta=Delta
    )
    
    print(f"Initial b-values (mean): {jnp.mean(initial_acq.bvalues):.2e}")
    
    # Calculate Initial Loss
    # We need to construct the acq variable dict manually for the loss function check
    trainable_acq = {
        'bvalues': initial_acq.bvalues,
        'gradient_directions': initial_acq.gradient_directions
    }
    static_acq = {'delta': delta, 'Delta': Delta}
    
    initial_loss = d_optimality_loss(trainable_acq, static_acq, model, target_params)
    print(f"Initial Negative LogDet FIM: {initial_loss:.4f}")
    
    # 3. Run Optimization
    print("\nRunning Optimization...")
    # b_max = 10,000 s/mm^2 = 10e9 s/m^2
    optimized_acq = optimize_protocol(
        initial_acq,
        model,
        target_params,
        n_steps=50,
        learning_rate=0.1, # Now using scaled b-values (order 1-10) and vecs (order 1)
        b_max=10e9
    )
    
    # Wait, simple gradient descent with one LR for both bvals (scale 1e9) and vecs (scale 1) is bad.
    # We should probably separate LRs or normalize variables.
    # But let's run it and see. If it fails, I'll update the OED code to accept separate LRs.
    
    print(f"\nOptimization Finished.")
    print(f"Optimized b-values (mean): {jnp.mean(optimized_acq.bvalues):.2e}")
    print(f"Optimized b-values (min): {jnp.min(optimized_acq.bvalues):.2e}")
    print(f"Optimized b-values (max): {jnp.max(optimized_acq.bvalues):.2e}")
    
    # Final Loss
    trainable_acq_opt = {
        'bvalues': optimized_acq.bvalues,
        'gradient_directions': optimized_acq.gradient_directions
    }
    final_loss = d_optimality_loss(trainable_acq_opt, static_acq, model, target_params)
    print(f"Final Negative LogDet FIM: {final_loss:.4f}")
    
    if final_loss < initial_loss:
        print("\nSUCCESS: Loss decreased (Information increased).")
    else:
        print("\nFAILURE: Loss did not decrease.")

if __name__ == "__main__":
    main()
