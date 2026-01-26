
import os
import sys
import time
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.io.wand import WANDLoader
from dmipy_jax.cylinder import C2Cylinder, C1Stick
from dmipy_jax.gaussian import G1Ball, G2Zeppelin
from dmipy_jax.distributions.distributions import DD1Gamma

def run_wand_comparison():
    print("Initiating WAND Comparison: AxCaliber vs ActiveAx vs ExCaliber")
    
    # 1. Load WAND Data
    print("Loading WAND Data...")
    # Adjust base_path if needed, assuming user has it in standard data dir
    loader = WANDLoader() 
    try:
        # Fetching might require GIN auth or it's public. Assuming public/installed.
        # Fallback to local if fetch fails/not configured.
        pass
    except:
        pass
        
    try:
        data_dict = loader.load_axcaliber_data(roi_slice=(slice(40, 50), slice(40, 50), 30))
    except FileNotFoundError:
        print("Data not found. Attempting fetch...")
        try:
            loader.fetch_data()
            data_dict = loader.load_axcaliber_data(roi_slice=(slice(40, 50), slice(40, 50), 30))
        except Exception as e:
            print(f"Failed to load WAND data: {e}")
            return

    data = data_dict['data']
    bvals = data_dict['bvals']
    bvecs = data_dict['bvecs']
    big_delta = data_dict['big_delta']
    small_delta = data_dict['small_delta']
    affine = data_dict['affine']
    
    print(f"Data Shape: {data.shape}")
    print(f"Max Gradient: {300 if jnp.max(bvals) > 10000 else 'Unknown'} mT/m equivalent")
    
    # Normalize
    b0_mask = bvals < 50
    b0_mean = jnp.mean(data[..., b0_mask], axis=-1, keepdims=True)
    b0_mean = jnp.maximum(b0_mean, 1e-6)
    data_norm = data / b0_mean
    
    # Flatten ROI for processing
    n_voxels = data_norm.shape[0] * data_norm.shape[1]
    signals = data_norm.reshape(n_voxels, -1)
    
    # 2. Define Models
    
    # A. AxCaliber (Standard)
    # Gamma Cylinder + Zeppelin (Linked Par)
    cylinder = C2Cylinder()
    zeppelin = G2Zeppelin()
    gamma_dist = DD1Gamma(Nsteps=15)
    
    def axcaliber_fn(params, bvals, bvecs, big, small):
        # [alpha, beta, f_intra, theta, phi] (Simplest)
        alpha = params[0]
        beta = params[1] * 1e-6
        f_intra = params[2]
        
        theta, phi = 1.57, 0.0 # Fixed for now or fitted
        mu = jnp.array([theta, phi])
        lambda_par = 1.7e-9
        
        # Intra
        radii, pdf = gamma_dist(alpha=alpha, beta=beta)
        d = 2*radii
        def sig_d(d_val):
             return cylinder(bvals, bvecs, mu=mu, lambda_par=lambda_par, diameter=d_val, big_delta=big, small_delta=small)
        S_intra = jnp.dot(pdf/jnp.sum(pdf), jax.vmap(sig_d)(d))
        
        # Extra
        S_extra = zeppelin(bvals, bvecs, mu=mu, lambda_par=lambda_par, lambda_perp=1.7e-9 * 0.3)
        
        return f_intra * S_intra + (1 - f_intra) * S_extra

    # B. ActiveAx (Alexander 2010)
    # Optimized for Index of Diameter.
    # Usually Cylinder + Zeppelin/Dot.
    # Key difference is PROTOCOL (which we assume WAND covers) and Model Constraints.
    # ActiveAx often assumes a single minimal radius index (Mean Diameter) rather than full distribution?
    # Or uses a distribution but focuses on specific shells.
    # Let's implement it as: Single Radius Cylinder + Zeppelin (Minimal Model).
    # Or Gamma Distributed but with fixed shape?
    # Let's use Single Radius for contrast.
    
    def activeax_fn(params, bvals, bvecs, big, small):
        # [diameter, f_intra]
        diameter = params[0] * 1e-6
        f_intra = params[1]
        
        theta, phi = 1.57, 0.0
        mu = jnp.array([theta, phi])
        lambda_par = 1.7e-9
        
        S_intra = cylinder(bvals, bvecs, mu=mu, lambda_par=lambda_par, diameter=diameter, big_delta=big, small_delta=small)
        S_extra = zeppelin(bvals, bvecs, mu=mu, lambda_par=lambda_par, lambda_perp=1.7e-9 * 0.3)
        
        return f_intra * S_intra + (1 - f_intra) * S_extra

    # C. ExCaliber (Cyl + Zepp + Iso) (As implemented before)
    ball = G1Ball()
    def excaliber_fn(params, bvals, bvecs, big, small):
        # [alpha, beta, f_intra, f_iso]
        alpha = params[0]
        beta = params[1] * 1e-6
        f_intra = params[2]
        f_iso = params[3]
        
        theta, phi = 1.57, 0.0
        mu = jnp.array([theta, phi])
        lambda_par = 1.7e-9
        
        radii, pdf = gamma_dist(alpha=alpha, beta=beta)
        d = 2*radii
        def sig_d(d_val):
             return cylinder(bvals, bvecs, mu=mu, lambda_par=lambda_par, diameter=d_val, big_delta=big, small_delta=small)
        S_intra = jnp.dot(pdf/jnp.sum(pdf), jax.vmap(sig_d)(d))
        
        S_extra = zeppelin(bvals, bvecs, mu=mu, lambda_par=lambda_par, lambda_perp=1.7e-9 * 0.3)
        S_iso = ball(bvals, lambda_iso=3.0e-9)
        
        return f_intra * S_intra + (1.0 - f_intra - f_iso) * S_extra + f_iso * S_iso
        
    # 3. Fitting Loop
    
    # Generic Fitter
    def fit_model(model_fn, init_params, name):
        print(f"Fitting {name}...")
        start_t = time.time()
        
        @jax.jit
        def loss(p, sig):
             pred = model_fn(p, bvals, bvecs, big_delta, small_delta)
             return jnp.mean((pred - sig)**2)
             
        optimizer = optax.adam(0.01)
        
        @jax.jit
        def step(carry, sig):
            p, state = carry
            l, grads = jax.value_and_grad(loss)(p, sig)
            updates, state = optimizer.update(grads, state, p)
            p_new = optax.apply_updates(p, updates)
            # Simple absolute clipping for stability
            p_new = jnp.abs(p_new) 
            return (p_new, state), l
            
        # Vmap over voxels? 
        # For simplicity in comparison script, loop over a few voxels or vmap small batch.
        # Let's vmap.
        
        def fit_voxel(sig):
            state = optimizer.init(init_params)
            (pf, _), _ = jax.lax.scan(lambda c, _: step(c, sig), (init_params, state), jnp.arange(150))
            return pf
            
        results = jax.vmap(fit_voxel)(signals)
        
        dur = time.time() - start_t
        print(f"  {name} finished in {dur:.2f}s")
        return results

    # Run Fits
    # AxCaliber: [alpha=4, beta=0.5, f=0.5]
    res_ax = fit_model(axcaliber_fn, jnp.array([4.0, 0.5, 0.5]), "AxCaliber")
    
    # ActiveAx: [diameter=2.0, f=0.5]
    res_aa = fit_model(activeax_fn, jnp.array([2.0, 0.5]), "ActiveAx")
    
    # ExCaliber: [alpha=4, beta=0.5, f=0.5, f_iso=0.1]
    res_ex = fit_model(excaliber_fn, jnp.array([4.0, 0.5, 0.5, 0.1]), "ExCaliber")
    
    # 4. Analysis / comparison
    # Mean Diameter Estimation
    # Aax: direct
    # Ax/Ex: mean of Gamma
    
    # AxCaliber Mean D
    ax_mean = 2 * res_ax[:, 0] * res_ax[:, 1] # alpha * beta (um)
    
    # ActiveAx Mean D
    aa_mean = res_aa[:, 0] # diameter (um)
    
    # ExCaliber Mean D
    ex_mean = 2 * res_ex[:, 0] * res_ex[:, 1]
    
    print("\n--- RESULTS SUMMARY (Mean across ROI) ---")
    print(f"AxCaliber Mean Diameter: {jnp.mean(ax_mean):.2f} um")
    print(f"ActiveAx Mean Diameter:  {jnp.mean(aa_mean):.2f} um")
    print(f"ExCaliber Mean Diameter: {jnp.mean(ex_mean):.2f} um")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.hist(ax_mean, bins=20, alpha=0.7, label='AxCaliber')
    plt.legend(); plt.title("AxCaliber Dia")
    
    plt.subplot(132)
    plt.hist(aa_mean, bins=20, alpha=0.7, color='orange', label='ActiveAx')
    plt.legend(); plt.title("ActiveAx Dia")
    
    plt.subplot(133)
    plt.hist(ex_mean, bins=20, alpha=0.7, color='green', label='ExCaliber')
    plt.legend(); plt.title("ExCaliber Dia")
    
    plt.savefig("wand_comparison_results.png")
    print("Comparison plot saved.")

if __name__ == "__main__":
    run_wand_comparison()
