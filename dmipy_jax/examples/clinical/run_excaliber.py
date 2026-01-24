
import os
import sys
import jax
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import optax
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dmipy_jax.io.multi_te import MultiTELoader
from dmipy_jax.cylinder import C2Cylinder
from dmipy_jax.gaussian import G1Ball, G2Zeppelin
from dmipy_jax.distributions.distributions import DD1Gamma
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme

def run_excaliber():
    print("Initializing ExCaliber (Cyl+Zepp+Iso) Example...")
    
    # 1. Load Data (Sub-023 Multi-TE)
    print("Loading Multi-TE Data for sub_023...")
    base_path = os.path.join(os.path.dirname(__file__), '../../../data/sub_023')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    loader = MultiTELoader(base_path=base_path, subject='sub_023')
    tes = loader.get_available_tes()
    
    if not tes:
        print("No data found.")
        return

    # Load and Concatenate
    # (Same loading logic as AxCaliber)
    all_data = []
    all_bvals = []
    all_bvecs = []
    all_big_deltas = []
    all_small_deltas = []
    
    slice_x, slice_y, slice_z = 33, 54, 30 
    
    for te in tes:
        if float(te) == 0: continue
        data, bvals, bvecs, protocol = loader.load_data(te)
        if protocol['Delta'] == 0: continue
        
        bvals = bvals * 1e6 # s/mm2 -> s/m2
        
        # Normalize per shell
        b0_mask = bvals < 50
        b0_val = jnp.mean(data[..., b0_mask], axis=-1, keepdims=True)
        b0_val = jnp.where(b0_val == 0, 1.0, b0_val)
        data_norm = data / b0_val
        data_norm = jnp.nan_to_num(data_norm)
        
        voxel_signal = data_norm[slice_x, slice_y, slice_z, :]
        
        all_data.append(voxel_signal)
        all_bvals.append(bvals)
        all_bvecs.append(bvecs)
        n_dwis = len(bvals)
        all_big_deltas.append(jnp.full((n_dwis,), protocol['Delta']))
        all_small_deltas.append(jnp.full((n_dwis,), protocol['delta']))

    full_signal = jnp.concatenate(all_data, axis=0)
    full_bvals = jnp.concatenate(all_bvals, axis=0)
    full_bvecs = jnp.concatenate(all_bvecs, axis=0)
    full_big_delta = jnp.concatenate(all_big_deltas, axis=0)
    full_small_delta = jnp.concatenate(all_small_deltas, axis=0)
    
    print(f"Total samples: {len(full_signal)}")
    
    # 2. Define ExCaliber Model
    # Cylinder (Intra) + Zeppelin (Extra) + Ball (Iso)
    
    # Components
    # We instantiate them to access their call methods
    cylinder = C2Cylinder()   # Intra: Distributed diameter
    zeppelin = G2Zeppelin()   # Extra: Anisotropic hindered
    ball = G1Ball()           # Iso: Free water (CSF)
    
    # Distribution
    gamma_dist = DD1Gamma(Nsteps=20)
    
    # Constants
    lambda_iso = 3.0e-9 # Free water diffusivity
    
    def excaliber_model(params, bvals, bvecs, big_delta, small_delta):
        # Unpack params
        # P = [theta, phi, lambda_par, lambda_perp, alpha, beta, f_intra, f_iso]
        
        theta = params['theta']
        phi = params['phi']
        mu = jnp.array([theta, phi])
        
        lambda_par = params['lambda_par']     # Intra/Extra parallel
        lambda_perp = params['lambda_perp']   # Extra perpendicular (Zeppelin)
        
        alpha = params['alpha']
        beta = params['beta']
        
        f_intra = params['f_intra']
        f_iso = params['f_iso']
        f_extra = 1.0 - f_intra - f_iso
        
        # 1. Intra: Gamma-Distributed Cylinder
        radii, pdf = gamma_dist(alpha=alpha, beta=beta)
        diameters = 2 * radii
        
        # vmap over diameters
        def signal_intra_d(d):
            return cylinder(bvals, bvecs, 
                          mu=mu, 
                          lambda_par=lambda_par,
                          diameter=d,
                          big_delta=big_delta,
                          small_delta=small_delta)
                          
        signals_intra_d = vmap(signal_intra_d)(diameters)
        pdf_normalized = pdf / jnp.sum(pdf)
        S_intra = jnp.dot(pdf_normalized, signals_intra_d)
        
        # 2. Extra: Zeppelin
        S_extra = zeppelin(bvals, bvecs, 
                          mu=mu, 
                          lambda_par=lambda_par, # Assumption: Tortuosity affects Extra, but often fitted or linked?
                          # ExCaliber usually links lambda_par_intra = lambda_par_extra = lambda_inf?
                          # Let's assume linked.
                          lambda_perp=lambda_perp)
                          
        # 3. Iso: Ball
        S_iso = ball(bvals, lambda_iso=lambda_iso)
        
        # Combine
        return f_intra * S_intra + f_extra * S_extra + f_iso * S_iso

    # 3. Fitting
    print("\nFitting ExCaliber Model to Voxel...")
    
    # Loss Function
    @jax.jit
    @jax.value_and_grad
    def loss_fn(p_unconstrained):
        # p_unconstrained: [log_alpha, log_beta_um, logit_f_intra, logit_f_iso, logit_perp_ratio]
        
        # Transform back to constrained space
        # alpha > 1.1 prevents Gamma pdf singularity at 0
        alpha = jnp.exp(p_unconstrained[0]) + 1.1
        beta_si = (jnp.exp(p_unconstrained[1]) + 0.01) * 1e-6
        
        f_intra = jax.nn.sigmoid(p_unconstrained[2])
        f_iso = jax.nn.sigmoid(p_unconstrained[3])
        
        # Ensure sum f <= 1 constraint?
        # Softmax is better for multiclass, but let's try scaling.
        # Or let them be independent and scale.
        sum_f = f_intra + f_iso
        scale = jnp.maximum(sum_f + 1e-6, 1.0)
        # Assuming f_extra = 1 - f_intra - f_iso. If sum > 1, f_extra < 0.
        # Let's use softmax logic for 3 compartments?
        # logits -> [f_intra, f_iso, f_extra]
        # But we only have 2 params.
        # Let's keep sigmoid and clip sum?
        # Better: parametrization:
        # f_intra = sigmoid(p1)
        # f_iso = sigmoid(p2) * (1 - f_intra)
        
        f_intra = jax.nn.sigmoid(p_unconstrained[2])
        f_iso = jax.nn.sigmoid(p_unconstrained[3]) * (1.0 - f_intra)
        
        perp_ratio = jax.nn.sigmoid(p_unconstrained[4])
        
        p_dict = {
            'theta': 1.57, 'phi': 0.0,
            'lambda_par': 1.7e-9,
            'alpha': alpha,
            'beta': beta_si,
            'f_intra': f_intra,
            'f_iso': f_iso,
            'lambda_perp': 1.7e-9 * perp_ratio
        }
        
        pred = excaliber_model(p_dict, full_bvals, full_bvecs, full_big_delta, full_small_delta)
        return jnp.mean((pred - full_signal)**2)

    # Init: [alpha=4, beta=0.5um, f_intra=0.5, f_iso=0.1, perp_ratio=0.3]
    # Inverse transform for init
    alpha_init = 4.0
    beta_init = 0.5
    f_intra_init = 0.5
    f_iso_init = 0.1
    perp_init = 0.3
    
    p_init = jnp.array([
        jnp.log(alpha_init - 1.1),
        jnp.log(beta_init - 0.01),
        0.0, # sigmoid(0) = 0.5. WE want 0.5. ok.
        -2.0, # sigmoid(-2) ~ 0.12. f_iso = 0.12 * 0.5 = 0.06. 
        # precise inverse:
        # f_intra = 0.5 -> logit(0.5) = 0.
        # f_iso_target = 0.1
        # f_iso = sigmoid(x) * (1 - 0.5) = 0.1 => sigmoid(x) = 0.2 => x = log(0.2/0.8) = log(0.25) = -1.38
        jnp.log(0.1 / (1.0 - 0.1)) # logit(0.1) approx for perp
    ])
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(p_init)
    params = p_init
    
    print(f"Initial Params (Transformed): {params}")
    
    loss_history = []
    for i in range(250):
        loss, grads = loss_fn(params)
        loss_history.append(loss)
        
        if jnp.isnan(loss):
            print("NaN Loss!")
            break
            
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if i % 20 == 0:
            print(f"Iter {i}: Loss={loss:.6f}")
            
    # Transform back for display
    final_alpha = jnp.exp(params[0]) + 0.1
    final_beta_um = jnp.exp(params[1]) + 0.01
    final_f_intra = jax.nn.sigmoid(params[2])
    final_f_iso = jax.nn.sigmoid(params[3]) * (1.0 - final_f_intra)
    final_perp_ratio = jax.nn.sigmoid(params[4])
    
    print("\nFitted ExCaliber Parameters:")
    print(f"  alpha: {final_alpha:.4f}")
    print(f"  beta_um: {final_beta_um:.4f}")
    print(f"  f_intra: {final_f_intra:.4f}")
    print(f"  f_iso: {final_f_iso:.4f}")
    print(f"  perp_ratio: {final_perp_ratio:.4f}")
        
    fit_alpha = final_alpha
    fit_beta_um = final_beta_um
    mean_d = 2 * fit_alpha * fit_beta_um
    print(f"\nEstimated Mean Axon Diameter: {mean_d:.4f} microns")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("ExCaliber Optimization")
    plt.xlabel("Iter")
    plt.ylabel("Loss")
    
    # Dist
    radii, pdf = gamma_dist(alpha=fit_alpha, beta=fit_beta_um*1e-6)
    plt.subplot(1, 2, 2)
    plt.plot(radii*2*1e6, pdf)
    plt.title(f"Axon Diameter Distribution\nMean={mean_d:.2f}um")
    plt.xlabel("Diameter (um)")
    
    plt.savefig("excaliber_results.png")
    print("Saved excaliber_results.png")

if __name__ == "__main__":
    run_excaliber()
