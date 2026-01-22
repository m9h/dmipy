
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import optimistix as optx
import time
from pathlib import Path
from dmipy_jax.signal_models.ivim import IVIM
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme
from dmipy_jax.fitting.optimization import VoxelFitter
from dmipy_jax.algebra.initializers import segmented_ivim_init

# Configuration
# AAPM Challenge usually has diffusion values in mm^2/s or 10^-3 mm^2/s
class AAPMConfig:
    # Typical b-values from literature for this challenge
    bvalues = jnp.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800]) * 1e6 # s/m^2? No, input usually s/mm^2
    # Dmipy uses SI units (s/m^2). 800 s/mm^2 = 800 * 1e6 s/m^2.
    bvalues_si = jnp.array([0, 10, 20, 30, 50, 80, 100, 200, 400, 800]) * 1e6
    
    # Image Size
    shape = (50, 50)
    
    # Ground Truth Ranges (SI Units)
    D_tissue_range = (0.5e-9, 2.5e-9)
    D_pseudo_range = (5e-9, 50e-9)
    f_range = (0.05, 0.4)

def generate_victre_proxy(shape, key):
    """
    Generates a proxy phantom resembling breast tissue structures (VICTRE).
    Mixture of fatty (low diffusion?) and fibroglandular (higher diffusion/perfusion).
    """
    nx, ny = shape
    X, Y = jnp.meshgrid(jnp.linspace(-1, 1, nx), jnp.linspace(-1, 1, ny))
    R = jnp.sqrt(X**2 + Y**2)
    
    # 1. Background (Fatty Tissue)
    # Low D_tissue, Low f? 
    # Actually Fat has very low D (~0.3e-9) in DWI but chemical shift artifacts.
    # Let's assume suppressed fat or just "Background Tissue".
    d_bg = 0.5e-9
    dp_bg = 10e-9
    f_bg = 0.05
    
    # 2. Fibroglandular Tissue (Blobs)
    mask_gland = (R < 0.7) & (R > 0.2) & (jnp.sin(5*X) * jnp.cos(5*Y) > 0)
    d_gland = 1.8e-9
    dp_gland = 40e-9
    f_gland = 0.25
    
    # 3. Tumor / Lesion (High f, Low D?)
    # Malignant tumors often have restricted diffusion (Low D) and High Perfusion?
    # Or cellularity -> Low D.
    mask_tumor = jnp.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1) > 0.5
    d_tumor = 0.9e-9
    dp_tumor = 20e-9
    f_tumor = 0.15
    
    # Compose
    D_map = jnp.full(shape, d_bg)
    Dp_map = jnp.full(shape, dp_bg)
    f_map = jnp.full(shape, f_bg)
    
    D_map = jnp.where(mask_gland, d_gland, D_map)
    Dp_map = jnp.where(mask_gland, dp_gland, Dp_map)
    f_map = jnp.where(mask_gland, f_gland, f_map)
    
    D_map = jnp.where(mask_tumor, d_tumor, D_map)
    Dp_map = jnp.where(mask_tumor, dp_tumor, Dp_map)
    f_map = jnp.where(mask_tumor, f_tumor, f_map)
    
    return jnp.stack([D_map, Dp_map, f_map], axis=-1)

class IVIMNetwork(eqx.Module):
    mlp: eqx.nn.MLP
    
    def __init__(self, key, n_inputs):
        self.mlp = eqx.nn.MLP(
            in_size=n_inputs, out_size=3, width_size=64, depth=3,
            activation=jax.nn.relu, key=key
        )
            
    def __call__(self, x):
        out = self.mlp(x)
        d1 = jax.nn.softplus(out[0]) * 1e-9
        d2 = jax.nn.softplus(out[1]) * 1e-9
        f_val = jax.nn.sigmoid(out[2])
        return jnp.stack([d1, d2, f_val])

def run_benchmark():
    print("=== AAPM IVIM Challenge Benchmark (Proxy) ===")
    
    # 1. Setup Data
    bvals = AAPMConfig.bvalues_si
    # Isotropic
    grad = jnp.zeros((len(bvals), 3))
    grad = grad.at[:, 0].set(1.0)
    acq = SimpleAcquisitionScheme(bvalues=bvals, gradient_directions=grad)
    
    key = jax.random.PRNGKey(2024)
    gt_params = generate_victre_proxy(AAPMConfig.shape, key)
    
    # Simulate
    ivim = IVIM()
    def sim(p): return ivim(bvals, grad, D_tissue=p[0], D_pseudo=p[1], f=p[2])
    
    clean_sig = jax.vmap(jax.vmap(sim))(gt_params)
    
    # Noise (SNR 20 at b0)
    # Signal amplitude ~ 1.0 (S0=1)
    sigma = 1.0 / 20.0
    k1, k2 = jax.random.split(key)
    n1 = jax.random.normal(k1, clean_sig.shape) * sigma
    n2 = jax.random.normal(k2, clean_sig.shape) * sigma
    noisy_sig = jnp.sqrt((clean_sig + n1)**2 + n2**2)
    
    print(f"Data Generated. Shape {noisy_sig.shape}, SNR=20")
    
    results = {}
    
    # 2. Method A: Neural Network (Prior)
    print("\n[Method A] Training Neural Network...")
    start = time.time()
    
    # Train logic
    key_net, k_train = jax.random.split(key)
    net = IVIMNetwork(key_net, len(bvals))
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, opt_state, p, s):
        def loss(m):
            pred = jax.vmap(m)(s)
            scales = jnp.array([1e9, 1e9, 1.0])
            return jnp.mean(((pred - p)*scales)**2)
        v, g = eqx.filter_value_and_grad(loss)(model)
        u, s_new = opt.update(g, opt_state, model)
        return eqx.apply_updates(model, u), s_new, v

    # Generator
    batch_size = 512
    min_v = jnp.array([0.1e-9, 1e-9, 0.0])
    max_v = jnp.array([3.0e-9, 100e-9, 0.5]) # Ranges
    
    for i in range(2000):
        # On fly batch
        k_batch, k_n = jax.random.split(jax.random.fold_in(k_train, i))
        p_b = min_v + jax.random.uniform(k_batch, (batch_size, 3)) * (max_v - min_v)
        s_b = jax.vmap(sim)(p_b)
        # Add noise
        s_b = jnp.sqrt((s_b + jax.random.normal(k_n, s_b.shape)*sigma)**2 + sigma**2) # Rice approx
        
        net, opt_state, _ = train_step(net, opt_state, p_b, s_b)
        
    # Infer
    flat_sig = noisy_sig.reshape(-1, len(bvals))
    pred_nn = jax.vmap(net)(flat_sig).reshape(AAPMConfig.shape + (3,))
    results['NN'] = pred_nn
    print(f"NN Time: {time.time() - start:.2f}s")
    
    # 3. Method B: Voxel-wise (Least Squares)
    print("\n[Method B] Voxel-wise Fitting...")
    start = time.time()
    
    # Use generic Fitter? Or custom loop for speed.
    # Let's use custom simple LM loop or scipy?
    # Use optax.least_squares like in demo
    
    def resid(p_scaled, args):
        s_obs = args[0]
        # p_scaled unconstrained
        d = jax.nn.softplus(p_scaled[0]) * 1e-9
        dp = jax.nn.softplus(p_scaled[1]) * 1e-9
        f = jax.nn.sigmoid(p_scaled[2])
        est = ivim(bvals, grad, D_tissue=d, D_pseudo=dp, f=f)
        return est - s_obs
        
    solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
    
    @jax.jit
    def fit_voxel(s, init):
        # optx.least_squares signature: fn, solver, y0, args, ...
        sol = optx.least_squares(resid, solver, init, args=(s,), max_steps=1000, throw=False)
        return sol.value
        
    # Init from NN prediction? Or mean.
    # Let's verify NN benefit. Init from "mean"
    p0 = jnp.array([1.0, 20.0, -2.0]) # unconstrained guess
    
    pred_ls_flat = jax.vmap(fit_voxel, in_axes=(0, None))(flat_sig, p0)
    
    # Convert back
    d_ls = jax.nn.softplus(pred_ls_flat[:, 0]) * 1e-9
    dp_ls = jax.nn.softplus(pred_ls_flat[:, 1]) * 1e-9
    f_ls = jax.nn.sigmoid(pred_ls_flat[:, 2])
    pred_ls = jnp.stack([d_ls, dp_ls, f_ls], axis=-1).reshape(AAPMConfig.shape + (3,))
    results['LS'] = pred_ls
    print(f"LS Time: {time.time() - start:.2f}s")

    # 4. Method C: Global TV
    print("\n[Method C] Global TV Regularization...")
    # Skip full impl here for brevity, assume LS + Post-process or reusing demo code?
    # In a full benchmark we SHOULD run it. 
    # Let's implement valid TV loop.
    
    @jax.jit
    def global_loss(p_map, data, lam):
        # p_map unconstrained
        # MSE
        def pixel_loss(p, s):
            d = jax.nn.softplus(p[0]) * 1e-9
            dp = jax.nn.softplus(p[1]) * 1e-9
            f = jax.nn.sigmoid(p[2])
            return jnp.mean((ivim(bvals, grad, D_tissue=d, D_pseudo=dp, f=f) - s)**2)
        
        mse = jnp.mean(jax.vmap(jax.vmap(pixel_loss))(p_map, data))
        
        # TV
        dx = p_map[1:, :, :] - p_map[:-1, :, :]
        dy = p_map[:, 1:, :] - p_map[:, :-1, :]
        tv = jnp.mean(jnp.abs(dx)) + jnp.mean(jnp.abs(dy))
        return mse + lam * tv
        
    # Init from LS result (inverse mapped) -> Warm start!
    # Reuse LS result
    p_ls_inv = jnp.stack([
        jnp.log(jnp.expm1(pred_ls[...,0]*1e9)),
        jnp.log(jnp.expm1(pred_ls[...,1]*1e9)),
        -jnp.log(1./pred_ls[...,2] - 1.)
    ], axis=-1)
    # Clip to avoid NaNs
    p_ls_inv = jnp.nan_to_num(p_ls_inv)
    
    opt_tv = optax.adam(0.01)
    state_tv = opt_tv.init(p_ls_inv)
    p_tv = p_ls_inv
    
    @jax.jit
    def step_tv(p, st, d):
        l, g = eqx.filter_value_and_grad(global_loss)(p, d, 0.005)
        u, st = opt_tv.update(g, st, p)
        return eqx.apply_updates(p, u), st, l
        
    for i in range(200):
        p_tv, state_tv, l = step_tv(p_tv, state_tv, noisy_sig)
        
    d_tv = jax.nn.softplus(p_tv[..., 0]) * 1e-9
    f_tv = jax.nn.sigmoid(p_tv[..., 2])
    pred_tv = jnp.stack([d_tv, jnp.zeros_like(d_tv), f_tv], axis=-1)
    results['TV-MSE'] = pred_tv
    print("TV (MSE) Done.")

    # 4b. Method D: Global TV (Rician Likelihood)
    print("\n[Method D] Global TV (Rician)...")
    
    @jax.jit
    def global_loss_rician(p_map, data, lam):
        sigma = 0.05 # Known sigma from data gen
        
        def rician_nll(p, s):
            d = jax.nn.softplus(p[0]) * 1e-9
            dp = jax.nn.softplus(p[1]) * 1e-9
            f = jax.nn.sigmoid(p[2])
            A = ivim(bvals, grad, D_tissue=d, D_pseudo=dp, f=f)
            
            # Besse function approx for gradients? jax.scipy.special.i0e
            # log(I0(z)) = log(i0e(z) * exp(z)) = log(i0e(z)) + z
            # z = A*s / sigma^2
            z = (A * s) / (sigma**2)
            log_i0 = jnp.log(jax.scipy.special.i0e(z)) + z
            
            # NLL = - (log_i0 - (A^2 + s^2)/(2sigma^2))
            ll = log_i0 - (A**2 + s**2) / (2 * sigma**2)
            return -jnp.mean(ll) # Minimize NLL

        loss_data = jnp.mean(jax.vmap(jax.vmap(rician_nll))(p_map, data))
        
        # TV
        dx = p_map[1:, :, :] - p_map[:-1, :, :]
        dy = p_map[:, 1:, :] - p_map[:, :-1, :]
        tv = jnp.mean(jnp.abs(dx)) + jnp.mean(jnp.abs(dy))
        return loss_data + lam * tv

    # Init from MSE result (Warm start)
    opt_ric = optax.adam(0.005)
    state_ric = opt_ric.init(p_tv)
    p_ric = p_tv
    
    @jax.jit
    def step_ric(p, st, d):
        l, g = eqx.filter_value_and_grad(global_loss_rician)(p, d, 0.005)
        u, st = opt_ric.update(g, st, p)
        return eqx.apply_updates(p, u), st, l
        
    for i in range(200):
        p_ric, state_ric, l = step_ric(p_ric, state_ric, noisy_sig)
        
    d_ric = jax.nn.softplus(p_ric[..., 0]) * 1e-9
    f_ric = jax.nn.sigmoid(p_ric[..., 2])
    pred_ric = jnp.stack([d_ric, jnp.zeros_like(d_ric), f_ric], axis=-1)
    results['TV-Rician'] = pred_ric
    print("TV (Rician) Done.")
    
    # 4c. Method E: Global TV (Algebraic Init)
    print("\n[Method E] Global TV (Algebraic Init)...")
    start_alg = time.time()
    
    # Algebraic Init
    # segmented_ivim_init works on last axis signals.
    # flat_sig shape (N_vox, N_b)
    # Returns (..., 3)
    p_alg_init_flat = jax.vmap(lambda s: segmented_ivim_init(bvals, s))(flat_sig)
    
    # Convert parameters to unconstrained for optimizer
    # d = softplus(p0) => p0 = inverse_softplus(d)
    # f = sigmoid(p2) => p2 = logit(f)
    # Helper for inverse mappings
    inv_softplus = lambda x: jnp.log(jnp.expm1(x))
    inv_sigmoid = lambda x: jnp.log(x / (1 - x))
    
    pad_inv = jnp.stack([
        inv_softplus(p_alg_init_flat[..., 0]*1e9), # D
        inv_softplus(p_alg_init_flat[..., 1]*1e9), # Dp
        inv_sigmoid(jnp.clip(p_alg_init_flat[..., 2], 1e-4, 0.9999)) # f
    ], axis=-1)
    
    pad_inv = jnp.nan_to_num(pad_inv).reshape(AAPMConfig.shape + (3,))
    
    # Optimization (MSE-TV)
    opt_alg = optax.adam(0.01)
    state_alg = opt_alg.init(pad_inv)
    p_alg = pad_inv
    
    @jax.jit
    def step_alg(p, st, d):
        # reuse global_loss from MSE section
        l, g = eqx.filter_value_and_grad(global_loss)(p, d, 0.005)
        u, st = opt_alg.update(g, st, p)
        return eqx.apply_updates(p, u), st, l
        
    for i in range(200):
        p_alg, state_alg, l = step_alg(p_alg, state_alg, noisy_sig)
        
    d_alg = jax.nn.softplus(p_alg[..., 0]) * 1e-9
    f_alg = jax.nn.sigmoid(p_alg[..., 2])
    pred_alg_tv = jnp.stack([d_alg, jnp.zeros_like(d_alg), f_alg], axis=-1)
    results['TV-Algebraic'] = pred_alg_tv
    print(f"TV (Algebraic) Done. Total time: {time.time() - start_alg:.2f}s")
    
    # 5. Analysis
    print("\n=== SCORECARD ===")
    for name, pred in results.items():
        # Mask Background for fair stats?
        # Calculate RMSE
        err = (pred - gt_params)
        mse_d = jnp.mean(err[..., 0]**2)
        mse_f = jnp.mean(err[..., 2]**2)
        print(f"[{name}] RMSE D: {jnp.sqrt(mse_d):.2e} | RMSE f: {jnp.sqrt(mse_f):.4f}")

if __name__ == "__main__":
    run_benchmark()
