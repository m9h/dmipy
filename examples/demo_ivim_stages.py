
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dmipy_jax.signal_models.ivim import IVIM
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme
import optax
import equinox as eqx
from dmipy_jax.fitting.optimization import VoxelFitter

# Enable 64-bit precision for stability
jax.config.update("jax_enable_x64", True)

def generate_ivim_phantom(shape=(20, 20)):
    """
    Generates a 2D phantom with a gradient of perfusion fraction (f) 
    and a step change in tissue diffusion (D_tissue).
    """
    nx, ny = shape
    
    # Gradient in f from 0.05 to 0.3
    f_map = jnp.linspace(0.05, 0.3, nx)[:, None] * jnp.ones((1, ny))
    
    # Step change in D_tissue (Top half high, Bottom half low)
    D_tissue_map = jnp.zeros(shape)
    D_tissue_map = D_tissue_map.at[:nx//2, :].set(1.0e-9)
    D_tissue_map = D_tissue_map.at[nx//2:, :].set(0.7e-9)
    
    # Constant D_pseudo
    # D* is typically 10x - 100x D_tissue (e.g. 10e-9 to 50e-9)
    D_pseudo_map = jnp.full(shape, 15.0e-9)
    
    gt_params = jnp.stack([D_tissue_map, D_pseudo_map, f_map], axis=-1)
    return gt_params

class IVIMNetwork(eqx.Module):
    mlp: eqx.nn.MLP
    
    def __init__(self, key, n_inputs):
            self.mlp = eqx.nn.MLP(
                in_size=n_inputs,
                out_size=3,
                width_size=64,
                depth=3,
                activation=jax.nn.relu,
                key=key
            )
            
    def __call__(self, x):
        out = self.mlp(x)
        d1 = jax.nn.softplus(out[0]) * 1e-9
        d2 = jax.nn.softplus(out[1]) * 1e-9
        f_val = jax.nn.sigmoid(out[2])
        return jnp.stack([d1, d2, f_val])

def main():
    print("=== IVIM Stages Demo ===")

    
    # 1. Setup Acquisition
    # IVIM needs low b-values to sensing perfusion
    bvals = jnp.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800, 1000])
    # Dummies for gradients (isotropic model)
    bvecs = jnp.zeros((len(bvals), 3))
    bvecs = bvecs.at[:, 0].set(1.0)
    
    acq = SimpleAcquisitionScheme(bvalues=bvals, gradient_directions=bvecs)
    
    # 2. Generate Phantom
    shape = (10, 10)
    gt_params = generate_ivim_phantom(shape)
    
    print(f"Phantom Shape: {shape}")
    print(f"B-values: {bvals}")
    
    # 3. Simulate Signal (noiseless)
    ivim_model = IVIM()
    
    # Map model over pixels
    # IVIM.__call__ signature: (bvals, gradient_directions, D_tissue, D_pseudo, f)
    # We need a wrapper to pass params as args or use vmap smartly.
    
    @jax.jit
    def simulate_pixel(params):
        return ivim_model(bvals, bvecs, D_tissue=params[0], D_pseudo=params[1], f=params[2])
    
    # vmap over x and y (flattened)
    gt_params_flat = gt_params.reshape(-1, 3)
    signal_noiseless = jax.vmap(simulate_pixel)(gt_params_flat)
    
    # Add Rician Noise
    key = jax.random.PRNGKey(42)
    noise_level = 0.05 # high noise for IVIM difficulty
    sigma = noise_level
    
    # Rician noise: sqrt((S + n1)^2 + n2^2)
    n1 = jax.random.normal(key, signal_noiseless.shape) * sigma
    n2 = jax.random.normal(key, signal_noiseless.shape) * sigma
    signal_noisy = jnp.sqrt((signal_noiseless + n1)**2 + n2**2)
    
    print("Signal simulated with Rician noise.")
    
    # 4. Stage 1: Voxel-wise Fit
    print("\n--- Stage 1: Voxel-wise Fit (L-BFGS-B) ---")
    
    # Define generic model func for Fitter
    # func(params, acq) -> signal
    def model_func_fit(params, acq):
        # params: [D, D*, f]
        return ivim_model(acq.bvalues, acq.gradient_directions, 
                          D_tissue=params[0], D_pseudo=params[1], f=params[2])
    
    # Bounds and Scales
    # D_tissue: 0 to 3e-9. Scale: 1e9 -> [0, 3]
    # D_pseudo: 0 to 100e-9. Scale: 1e9 -> [0, 100]
    # f: 0 to 1. Scale: 1 -> [0, 1]
    
    ranges = [
        (0.0, 3e-9),    # D_tissue
        (0.0, 100e-9),  # D_pseudo
        (0.0, 1.0)      # f
    ]
    scales = [1e-9, 1e-9, 1.0]
    
    fitter = VoxelFitter(model_func_fit, ranges, scales=scales)
    
    # Initial guess
    # Start with reasonable assumptions
    init_guess = jnp.array([1.0e-9, 10.0e-9, 0.1])
    
    # Vmap fitter over all voxels
    # fit(data, acq, init_params)
    
    # We need to replicate init_guess for all voxels
    init_guess_batch = jnp.tile(init_guess, (len(gt_params_flat), 1))
    
    # Run fit
    # We wrap in jit to compile the loop
    @jax.jit
    def run_fitting(data_flat, init_flat):
        return jax.vmap(fitter.fit, in_axes=(0, None, 0))(data_flat, acq, init_flat)
    
    print("Running fitting...")
    fitted_params_flat, state = run_fitting(signal_noisy, init_guess_batch)
    fitted_params = fitted_params_flat.reshape(shape[0], shape[1], 3)
    
    # Metrcs
    rmse_D = jnp.sqrt(jnp.mean((fitted_params[..., 0] - gt_params[..., 0])**2))
    rmse_f = jnp.sqrt(jnp.mean((fitted_params[..., 2] - gt_params[..., 2])**2))
    
    print(f"Fit Complete.")
    print(f"RMSE D_tissue: {rmse_D:.2e}")

    print(f"RMSE f: {rmse_f:.4f}")

    # --- Stage 2: Global TV Regularized Fit ---
    print("\n--- Stage 2: Global TV Regularized Fit (Adam) ---")
    
    # Define Total Variation Loss
    def total_variation_2d(params_map):
        # params_map: (H, W, C)
        # Compute gradients along H and W
        diff_h = params_map[1:, :, :] - params_map[:-1, :, :]
        diff_w = params_map[:, 1:, :] - params_map[:, :-1, :]
        
        # L1 norm of gradients
        tv = jnp.sum(jnp.abs(diff_h)) + jnp.sum(jnp.abs(diff_w))
        return tv

    # Global Loss Function
    def global_loss(params_map_scaled, data, acq, lambda_tv):
        # params_map_scaled: (H, W, 3) unconstrained
        # We need to constrain parameters for physical validity
        # D, D* > 0 (Softplus), f in [0, 1] (Sigmoid)
        
        # Mapping
        # indices: 0->D_tissue, 1->D_pseudo, 2->f
        D_tissue = jax.nn.softplus(params_map_scaled[..., 0]) * 1e-9
        D_pseudo = jax.nn.softplus(params_map_scaled[..., 1]) * 1e-9
        f = jax.nn.sigmoid(params_map_scaled[..., 2])
        
        # Reconstruct params for TV
        # We compute TV on the physical parameters or the unconstrained ones?
        # Usually physical is better for interpretability, but unconstrained is smoother for updates.
        # Let's use physical parameters for TV to encourage smoothness in the result.
        phys_map = jnp.stack([D_tissue, D_pseudo, f], axis=-1)
        
        # Calculate Signals
        # Vectorized call to IVIM
        # ivim(b, g, D, D*, f)
        # We need to broadcast acq over H, W
        
        # Reshape for vmap
        H, W, _ = phys_map.shape
        phys_flat = phys_map.reshape(-1, 3)
        
        # Reuse previous simulate_pixel (it was JITed, need to re-define or import)
        # Note: ivim_model is from outer scope
        
        def sim_fn(p):
            return ivim_model(acq.bvalues, acq.gradient_directions, 
                              D_tissue=p[0], D_pseudo=p[1], f=p[2])
        
        preds_flat = jax.vmap(sim_fn)(phys_flat)
        preds = preds_flat.reshape(H, W, -1)
        
        # Data Fitting Term (MSE)
        mse = jnp.mean((preds - data) ** 2)
        
        # TV Term
        # Weight TV for each parameter if scales differ significantly?
        # D ~ 1e-9, f ~ 0.1.
        # If we just sum, D's TV will be negligible (10^-18).
        # We MUST normalize the TV contribution.
        # Normalize by typical values: D_scale=1e9, f_scale=1.
        
        phys_map_norm = jnp.stack([
            D_tissue * 1e9,
            D_pseudo * 1e9,
            f
        ], axis=-1)
        
        tv_val = total_variation_2d(phys_map_norm)
        
        # Normalize TV by number of pixels to keep lambda invariant with resolution
        tv_val = tv_val / (H * W)
        
        return mse + lambda_tv * tv_val

    # Optimization Loop
    @jax.jit
    def update(params, opt_state, data, lambda_tv):
        loss, grads = jax.value_and_grad(global_loss)(params, data, acq, lambda_tv)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    # Initialize Parameters
    # Use the result from Stage 1 as initialization?
    # Or random/mean.
    # Stage 1 result is ideal start.
    # We need to inverse-map the fitted parameters to the unconstrained space.
    
    # Params: D, D*, f
    # Softplus^-1(y) = log(exp(y) - 1)
    # Sigmoid^-1(y) = log(y / (1 - y))
    
    # Handle Stage 1 outputs
    # fitted_params is (H, W, 3).
    # Replace NaNs with suitable defaults if any
    fitted_params = jnp.nan_to_num(fitted_params, nan=1e-9)
    
    # Be careful of boundary values (0 or 1).
    eps = 1e-12
    d1 = jnp.clip(fitted_params[..., 0], eps, None) / 1e-9
    d2 = jnp.clip(fitted_params[..., 1], eps, None) / 1e-9
    f_val = jnp.clip(fitted_params[..., 2], 1e-6, 1-1e-6)
    
    # Inv-Softplus: x = log(exp(y) - 1). For large y, approx y.
    # jnp.log(jnp.expm1(y))
    inv_sp_d1 = jnp.log(jnp.expm1(d1))
    inv_sp_d2 = jnp.log(jnp.expm1(d2))
    inv_sig_f = jnp.log(f_val / (1 - f_val))
    
    init_map = jnp.stack([inv_sp_d1, inv_sp_d2, inv_sig_f], axis=-1)
    
    # Optimizer
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(init_map)
    
    current_params = init_map
    lambda_tv = 0.001 * 0.1 # Heuristic scaling
    # MSE is approx (0.1)^2 = 0.01.
    # TV is approx 0.1 per pixel fit diff.
    # Need to tune lambda.
    
    print("Running Global Optimization...")
    signal_noisy_reshaped = signal_noisy.reshape(shape[0], shape[1], -1)
    
    for i in range(200):
        current_params, opt_state, loss_val = update(current_params, opt_state, signal_noisy_reshaped, lambda_tv)
        if i % 50 == 0:
            print(f"Iter {i}: Loss {loss_val:.2e}")
            
    # Final Result
    D_tissue_tv = jax.nn.softplus(current_params[..., 0]) * 1e-9
    f_tv = jax.nn.sigmoid(current_params[..., 2])
    
    rmse_D_tv = jnp.sqrt(jnp.mean((D_tissue_tv - gt_params[..., 0])**2))
    rmse_f_tv = jnp.sqrt(jnp.mean((f_tv - gt_params[..., 2])**2))
    

    print(f"Global Fit Complete.")
    print(f"RMSE D_tissue (TV): {rmse_D_tv:.2e}")
    print(f"RMSE f (TV): {rmse_f_tv:.4f}")

    # --- Stage 3: Amortized Inference (Neural Network) ---
    print("\n--- Stage 3: Amortized Inference (Neural Network) ---")
    
    # Trainer
    key = jax.random.PRNGKey(55)
    key_net, key_train = jax.random.split(key)
    net = IVIMNetwork(key_net, n_inputs=len(acq.bvalues))
    
    # Optimizer for NN
    opt_net = optax.adam(1e-3)
    opt_state_net = opt_net.init(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch_params, batch_signal):
        
        def loss_fn(m):
            # Predict params from signal
            # vmap over batch
            pred_params = jax.vmap(m)(batch_signal)
            
            # Loss: MSE on parameters (normalized?)
            # D fits are tiny (1e-9), f is 0.1.
            # We must normalize to balance loss!
            # Scale D by 1e9, f by 1.
            scales = jnp.array([1e9, 1e9, 1.0])
            diff = (pred_params - batch_params) * scales
            return jnp.mean(diff ** 2)
            
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt_net.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    # Training Loop - Generate data on fly
    print("Training Network...")
    n_steps = 1000
    batch_size = 256
    
    # Ranges for training data
    # D: 0-3e-9, D*: 0-100e-9, f: 0-0.5
    min_vals = jnp.array([0.1e-9, 1e-9, 0.0])
    max_vals = jnp.array([3.0e-9, 50e-9, 0.5])
    
    @jax.jit
    def get_batch(k):
        # Uniform sampling
        r = jax.random.uniform(k, (batch_size, 3)) # 0-1
        # Scale
        p = min_vals + r * (max_vals - min_vals)
        
        # Simulate Signal
        # Use simple simulation function
        def sim(param):
            return ivim_model(acq.bvalues, acq.gradient_directions, 
                              D_tissue=param[0], D_pseudo=param[1], f=param[2])
                              
        sig = jax.vmap(sim)(p)
        # Add Noise
        k1, k2 = jax.random.split(k) # careful with k splitting
        # Proper split in loop
        return p, sig

    # Training
    for step in range(n_steps):
        key_train, k_batch, k_noise = jax.random.split(key_train, 3)
        
        # Generator
        # Cannot use get_batch with randomness inside perfectly if not passing keys
        # We pass k_batch
        params_batch, signal_batch_clean = get_batch(k_batch)
        
        # Noise
        noise = jax.random.normal(k_noise, signal_batch_clean.shape) * 0.05 # Sigma=0.05
        # Rician approximation or just Gaussian for training?
        # Gaussian is fine for NN usually, but Rician bias might matter.
        # Let's add Rician
        n1 = noise
        n2 = jax.random.normal(k_batch, signal_batch_clean.shape) * 0.05
        signal_batch = jnp.sqrt((signal_batch_clean + n1)**2 + n2**2)
        
        net, opt_state_net, loss = train_step(net, opt_state_net, params_batch, signal_batch)
        
        if step % 200 == 0:
            print(f"Step {step}: Loss {loss:.4f}")
            
    # Inference on Phantom
    print("Inferring on Phantom...")
    # signal_noisy is (N_voxels, N_b)
    # vmap net
    pred_params_nn = jax.vmap(net)(signal_noisy)
    pred_params_nn = pred_params_nn.reshape(shape[0], shape[1], 3)
    
    rmse_D_nn = jnp.sqrt(jnp.mean((pred_params_nn[..., 0] - gt_params[..., 0])**2))
    rmse_f_nn = jnp.sqrt(jnp.mean((pred_params_nn[..., 2] - gt_params[..., 2])**2))
    
    print(f"Neural Fit Complete.")
    print(f"RMSE D_tissue (NN): {rmse_D_nn:.2e}")
    print(f"RMSE f (NN): {rmse_f_nn:.4f}")
    
    return fitted_params, current_params, pred_params_nn



if __name__ == "__main__":
    main()
