
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from dmipy_jax.inference.amortized import ZeppelinNetwork
from dmipy_jax.inference.ssft import SSFTTrainer
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme as JaxAcquisition
from dmipy_jax.signal_models.zeppelin import Zeppelin

def main():
    print("=== SSFT Verification Experiment ===")
    key = jax.random.PRNGKey(0)
    
    # 1. Synthetic Data Setup (Small Volume)
    shape = (10, 10, 5)
    n_meas = 30
    
    # Simple Block Structure
    # Background: low diffusivity
    # Box in center: high diffusivity
    
    lambda_par_true = jnp.ones(shape) * 1.0e-9
    lambda_par_true = lambda_par_true.at[3:7, 3:7, :].set(2.0e-9)
    
    # Random other params
    lambda_perp_true = jnp.ones(shape) * 0.2e-9
    mu_true = jnp.zeros(shape + (2,)) # theta, phi = 0
    
    # Acquisition
    bvals = jnp.linspace(0, 3000, n_meas)
    bvecs = jax.random.normal(key, (n_meas, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    acq = JaxAcquisition(bvals, bvecs)
    
    # Generate Clean Signal
    # Use vmap to handle volume parameters
    def gen_voxel(l_par, l_perp, m):
        mod = Zeppelin(lambda_par=l_par, lambda_perp=l_perp, mu=m)
        return mod(acq.bvalues, acq.gradient_directions)
        
    vmap_gen = jax.vmap(jax.vmap(jax.vmap(gen_voxel)))
    signal_clean = vmap_gen(lambda_par_true, lambda_perp_true, mu_true)
    
    # Add Noise
    signal_noisy = signal_clean + jax.random.normal(key, signal_clean.shape) * 0.05
    
    # 2. Network Initialization
    # Randomly initialized network (untrained)
    net_key, init_key = jax.random.split(key)
    network = ZeppelinNetwork(
        key=init_key, 
        n_input_measurements=n_meas, 
        width_size=32, 
        depth=2
    )
    
    # 3. Parameter Map before SSFT (Initial Guess)
    vmap_net = jax.vmap(jax.vmap(jax.vmap(network)))
    preds_init = vmap_net(signal_noisy)
    map_init = preds_init['lambda_par']
    
    # 4. Run SSFT
    trainer = SSFTTrainer(learning_rate=1e-3)
    
    print("Running SSFT...")
    # Lambda TV = 0.01
    network_tuned = trainer.fit(
        network, 
        signal_noisy, 
        acq, 
        lambda_tv=0.01, 
        epochs=50 # Fast training
    )
    
    # 5. Parameter Map after SSFT
    vmap_net_tuned = jax.vmap(jax.vmap(jax.vmap(network_tuned)))
    preds_tuned = vmap_net_tuned(signal_noisy)
    map_tuned = preds_tuned['lambda_par']
    
    # 6. Compare
    mse_init = jnp.mean((map_init - lambda_par_true)**2)
    mse_tuned = jnp.mean((map_tuned - lambda_par_true)**2)
    
    print(f"MSE Init: {mse_init:.2e}")
    print(f"MSE Tuned: {mse_tuned:.2e}")
    
    # Calculate TV of the maps
    def calc_tv(x):
        return sum([jnp.sum(jnp.abs(g)) for g in jnp.gradient(x)])
        
    tv_init = calc_tv(map_init)
    tv_tuned = calc_tv(map_tuned)
    
    print(f"TV Init: {tv_init:.2e}")
    print(f"TV Tuned: {tv_tuned:.2e}")
    
    # Assert improvement
    # Note: Since network is random init, MSE might improve just by fitting data.
    # But TV should definitely be reasonable.
    
    if mse_tuned < mse_init:
        print("SUCCESS: MSE improved after tuning.")
    else:
        print("WARNING: MSE did not improve (might need more training/better init).")
        
    # Plot center slice
    plt.figure(figsize=(10, 4))
    plt.subplot(1,3,1); plt.imshow(lambda_par_true[..., 2]); plt.title('Ground Truth')
    plt.subplot(1,3,2); plt.imshow(map_init[..., 2]); plt.title('Initial (Random)')
    plt.subplot(1,3,3); plt.imshow(map_tuned[..., 2]); plt.title('Tuned (TV)')
    plt.savefig("experiments/ssft_verification.png")
    print("Saved plot.")

if __name__ == "__main__":
    main()
