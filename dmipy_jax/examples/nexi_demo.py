
import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from jax import random

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dmipy_jax.io.multi_te import MultiTELoader
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin
from dmipy_jax.components.exchange import KargerExchange

def run_nexi_demo():
    print("==============================================")
    print("   NEXI (Neurite Exchange Imaging) Demo       ")
    print("   Model: Standard Model w/ Exchange (SMEX)   ")
    print("   Components: Stick (Intra) + Zeppelin (Extra)")
    print("==============================================")

    # 1. Setup Data
    # Try to load Multi-TE data
    base_path_guess = "/home/mhough/dev/dmipy/data/MultiTE/MTE-dMRI"
    subject = "sub-03"
    
    data_loaded = False
    
    # Placeholder variables
    all_signals = []
    all_bvals = []
    all_bvecs = []
    all_deltas = []
    all_Deltas = []
    
    try:
        loader = MultiTELoader(base_path_guess, subject)
        tes = loader.get_available_tes()
        print(f"Found Multi-TE dataset with TEs: {tes}")
        
        # Load all TEs
        for te_str in tes:
            # Parse TE string to get approximate Delta/delta?
            # Creating heuristics similar to MultiTELoader logic if needed, 
            # but for now we trust the loader or use metadata if available.
            # MultiTELoader from dmipy_jax.io.multi_te returns (data, bvals, bvecs, protocol_dict)
            d, b, v, protocol = loader.load_data(te_str)
            
            # Extract a single ROI to keep things fast
            # Center slice
            sx, sy, sz, _ = d.shape
            roi_signals = d[sx//2 : sx//2 + 5, sy//2 : sy//2 + 5, sz//2, :] # 5x5 ROI
            roi_signals = roi_signals.reshape(-1, roi_signals.shape[-1])
            
            # Normalize
            b0_mask = b < 50
            if jnp.sum(b0_mask) > 0:
                s0 = jnp.mean(roi_signals[:, b0_mask], axis=1, keepdims=True)
                roi_signals = roi_signals / jnp.maximum(s0, 1e-6)
            
            # Append
            all_signals.append(roi_signals)
            all_bvals.append(b)
            all_bvecs.append(v)
            
            # Use protocol fingerprint if available, or try parsing TE string
            # Protocol usually has 'delta' (ms) and 'Delta' (ms) if hardcoded, or s if dynamic.
            # multi_te.py PROTOCOL_FINGERPRINT has values in ms (e.g. 15.2, 25.2).
            # We need seconds.
            
            try:
                # If protocol dict exists and has values
                # Check units. Usually code returns raw values. dmipy usually expects seconds.
                # multi_te.py returns 15.2 ms.
                delta_val = protocol.get('delta', 15.2) * 1e-3
                Delta_val = protocol.get('Delta', 25.2) * 1e-3
                
                # If te_str implies specific Delta?
                # The loader logic might return generic protocol.
                # Let's rely on protocol if it looks dynamic, otherwise fallback to TE estimation for checks.
                
                delta_est = delta_val
                Delta_est = Delta_val
                
            except:
                # Fallback heuristics
                try:
                    te_val = float(te_str.replace('R1','').replace('R2','')) * 1e-3 # s
                except:
                    te_val = 0.1
                
                delta_est = 0.015 # 15ms
                Delta_est = te_val - delta_est - 0.005 # Rough
            
            all_deltas.append(jnp.full(b.shape, delta_est))
            all_Deltas.append(jnp.full(b.shape, Delta_est))
            
        data_loaded = True
        print(f"Loaded {len(tes)} TE shells.")
        
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load Multi-TE dataset: {e}")
        print("Falling back to Synthetic Data Generation...")
    
    # Synthetic Generation if needed
    if not data_loaded:
        print("Simulating SMEX signals...")
        # Define Ground Truth
        # f_stick = 0.6, f_zep = 0.4
        # D_intra = 2.0, D_extra = 1.0
        # Exchange time = 0.030 s (30ms) -> K_ie ~ 33 1/s
        
        # Acquisition: 3 shells (b=1000, 2000, 3000), 2 Deltas (30ms, 60ms)
        b_shells = jnp.array([1000, 2000, 3000]) * 1e6
        n_dirs = 12
        vecs = random.normal(random.PRNGKey(0), (n_dirs, 3))
        vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
        
        deltas = [0.010, 0.010]
        Deltas = [0.030, 0.060]
        
        sim_signals = []
        sim_bvals = []
        sim_bvecs = []
        sim_Deltas = []
        sim_deltas = []
        
        # Ground Truth Components
        # Model: Karger([Stick, Zeppelin])
        stick = C1Stick()
        zeppelin = G2Zeppelin()
        karger = KargerExchange([stick, zeppelin])
        
        # Params:
        # Stick: mu (2), lambda_par (1)
        # Zeppelin: mu (2), lambda_par (1), lambda_perp (1)
        # Fractions: f_stick (1) [f_zep implicit]
        # Exchange: tau_12 (1)
        
        # Shared orientation Z-axis
        theta, phi = 0.0, 0.0
        lambda_par = 2.5e-9 
        lambda_perp = 0.8e-9
        
        f_stick = 0.6
        tau_12 = 0.050 # 50ms exchange time
        
        # Pack params 
        # C1Stick: theta, phi, lambda_par
        # G2Zeppelin: theta, phi, lambda_par, lambda_perp
        # Fractions: f_stick
        # Exchange: tau_12
        gt_params = jnp.array([
            theta, phi, lambda_par,              # Stick
            theta, phi, lambda_par, lambda_perp, # Zeppelin
            f_stick,                             # Fraction
            tau_12                               # Exchange
        ])
        
        # Simulate for each Delta config
        for i in range(len(Deltas)):
            d = deltas[i]
            D = Deltas[i]
            
            b = jnp.kron(b_shells, jnp.ones(n_dirs))
            v = jnp.tile(vecs, (3, 1))
            
            acq = JaxAcquisition(
                bvalues=b,
                gradient_directions=v,
                delta=d,
                Delta=D
            )
            
            # Predict
            sig = karger.predict(gt_params, acq)
            
            # Add noise
            keys = random.split(random.PRNGKey(i), len(sig))
            noise = random.normal(random.PRNGKey(i+100), sig.shape) * 0.02
            sig_noisy = jnp.abs(sig + noise)
            
            # Create "Voxels" (just repeat GT for now with different noise?)
            # Let's make 20 voxels
            n_vox = 20
            batch_noise = random.normal(random.PRNGKey(i+200), (n_vox, len(sig))) * 0.02
            batch_sig = jnp.abs(sig + batch_noise)
            
            all_signals.append(batch_sig)
            all_bvals.append(b)
            all_bvecs.append(v)
            all_deltas.append(jnp.full(b.shape, d))
            all_Deltas.append(jnp.full(b.shape, D))
            
    # 2. Concatenate Data for Global Fitting
    # We need to stack everything.
    # Voxels are consistent across separate acquisitions?
    # In Multi-TE loader logic, if we extracted ROI, yes.
    # In Synth, yes.
    
    cat_signals = jnp.concatenate(all_signals, axis=1) # (N_vox, Total_Meas)
    cat_bvals = jnp.concatenate(all_bvals)
    cat_bvecs = jnp.concatenate(all_bvecs, axis=0)
    cat_deltas = jnp.concatenate(all_deltas)
    cat_Deltas = jnp.concatenate(all_Deltas)
    
    print(f"Total Measurements per Voxel: {cat_signals.shape[1]}")
    
    full_acq = JaxAcquisition(
        bvalues=cat_bvals,
        gradient_directions=cat_bvecs,
        delta=cat_deltas,
        Delta=cat_Deltas
    )
    
    # 3. Define Model
    stick = C1Stick()
    zeppelin = G2Zeppelin()
    model = KargerExchange([stick, zeppelin])
    
    # 4. Fit
    # Loss function
    @jax.jit
    def loss_fn(params, signal_target, acq):
        # Constrain params via softplus/sigmoid or clipping?
        # Params:
        # 0,1: theta, phi
        # 2: lambda_par_stick
        # 3,4: theta_z, phi_z (Should link to 0,1)
        # 5: lambda_par_zep (Should link to 2)
        # 6: lambda_perp_zep
        # 7: f_stick
        # 8: tau_12
        
        # We manually link parameters for SMEX
        # x are the optimization variables
        # Let's define x: [theta, phi, lambda_par, lambda_perp, f_stick, tau_12]
        
        theta, phi = params[0], params[1]
        l_par = params[2] # Bound > 0
        l_perp = params[3] # Bound > 0, < l_par
        f_int = params[4] # 0..1
        tau = params[5] # > 0
        
        # Reconstruct full params for KargerExchange.predict
        # Stick: t, p, l_par
        # Zep: t, p, l_par, l_perp
        # Frac: f_int
        # Exch: tau
        
        full_p = jnp.array([
            theta, phi, l_par,
            theta, phi, l_par, l_perp,
            f_int,
            tau
        ])
        
        pred = model.predict(full_p, acq)
        return jnp.mean((pred - signal_target)**2)

    optimizer = optax.adam(learning_rate=0.005)
    
    @jax.jit
    def fit_voxel(signal, init_p):
        opt_state = optimizer.init(init_p)
        
        def step(carry, i):
            p, state = carry
            l, grads = jax.value_and_grad(loss_fn)(p, signal, full_acq)
            updates, state = optimizer.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            
            # Constraints
            # l_par positive approx 3e-9 max
            new_p = new_p.at[2].set(jnp.clip(new_p[2], 0.1e-9, 3.5e-9))
            
            # l_perp positive, < l_par
            new_p = new_p.at[3].set(jnp.clip(new_p[3], 0.0, new_p[2] - 1e-10))
            
            # f_int 0..1
            new_p = new_p.at[4].set(jnp.clip(new_p[4], 0.01, 0.99))
            
            # tau > 0
            new_p = new_p.at[5].set(jnp.clip(new_p[5], 0.001, 1.0)) # 1ms to 1s
            
            return (new_p, state), l
            
        (final_p, _), _ = jax.lax.scan(step, (init_p, opt_state), jnp.arange(500))
        return final_p

    print("Fitting Voxels...")
    # Init: [0, 0, 2e-9, 1e-9, 0.5, 0.1]
    init_x = jnp.array([0.0, 0.0, 2.0e-9, 0.5e-9, 0.5, 0.1])
    
    fitted_x = jax.vmap(fit_voxel, in_axes=(0, None))(cat_signals, init_x)
    
    # 5. Visualize Results
    taus = fitted_x[:, 5] * 1000 # ms
    f_ints = fitted_x[:, 4]
    
    print(f"Mean Exchange Time: {jnp.mean(taus):.2f} ms")
    print(f"Mean Neurite Fraction: {jnp.mean(f_ints):.2f}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(taus, bins=10, color='skyblue', edgecolor='black')
    plt.title('Exchange Time Distribution')
    plt.xlabel('Tau (ms)')
    
    plt.subplot(1, 2, 2)
    plt.hist(f_ints, bins=10, color='lightgreen', edgecolor='black')
    plt.title('Neurite Fraction Distribution')
    plt.xlabel('f_intra')
    
    out_file = "nexi_results.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    run_nexi_demo()
