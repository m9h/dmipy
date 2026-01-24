
import os
import sys
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
from dmipy_jax.signal_models.sandi import SphereGPD
from dmipy_jax.components.exchange import KargerExchange
from dmipy_jax.io.connectome2 import load_connectome2_mri

def run_sandix_demo():
    print("==============================================")
    print("   SANDIX (SANDI with Exchange) Demo          ")
    print("   Model: SANDI + Exchange                    ")
    print("   Components: Stick (Neurite), Sphere (Soma),")
    print("               Zeppelin (Extra), Ball (CSF)   ")
    print("==============================================")

    # 1. Setup Data
    # Try loading Connectome 2.0
    print("Attempting to load Connectome 2.0 (sub-01)...")
    
    data_loaded = False
    
    try:
        data_dict = load_connectome2_mri(subject="sub-01", voxel_slice=(slice(45, 50), slice(45, 50), 30))
        dwi = data_dict['dwi']
        bvals = data_dict['bvals']
        bvecs = data_dict['bvecs']
        
        # Flatten
        n_x, n_y, n_z, n_dwis = dwi.shape
        data = dwi.reshape(-1, n_dwis)
        
        # Normalize
        b0_mask = bvals < 50
        b0 = jnp.mean(data[:, b0_mask], axis=1, keepdims=True)
        data = data / jnp.maximum(b0, 1e-6)
        
        # Acquisition
        # We assume standard approx timing if not in dict
        delta = 0.0129
        Delta = 0.0218
        
        acq = JaxAcquisition(
            bvalues=jnp.array(bvals),
            gradient_directions=jnp.array(bvecs),
            delta=delta,
            Delta=Delta
        )
        
        data_loaded = True
        print(f"Loaded {data.shape[0]} voxels.")
        
    except Exception as e:
        print(f"Could not load Connectome data: {e}")
        print("Falling back to Synthetic SANDIX Generation...")
        
    if not data_loaded:
        # Simulate SANDIX
        # Config: High b-values (up to 10k)
        b_shells = jnp.array([1000, 3000, 5000, 10000]) * 1e6
        n_dirs = 16
        vecs = jax.random.normal(jax.random.PRNGKey(0), (n_dirs, 3))
        vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
        
        bvals = jnp.kron(b_shells, jnp.ones(n_dirs))
        bvecs = jnp.tile(vecs, (4, 1))
        
        delta = 0.015
        Delta = 0.040 # Longer time to allow exchange effect?
        
        acq = JaxAcquisition(
            bvalues=bvals,
            gradient_directions=bvecs,
            delta=delta,
            Delta=Delta
        )
        
        # Components
        # [Stick, Sphere, Zeppelin, Ball]
        # BUT KargerExchange typically couples compartments.
        # Ball (CSF) is usually uncoupled or fast exchange? Usually uncoupled.
        # We will model Exchange between Stick (Neurite) and Zeppelin (Extra-neurite).
        # Soma (Sphere) might exchange with Extra-neurite? 
        # For simplicity, let's assume Neurite <-> Extra exchange only, treating Soma and Ball as isolated.
        # However, KargerExchange logic in current `dmipy_jax` couples ALL passed models fully unless exchange times are Inf.
        # If we pass all 4 to KargerExchange, we get a 4x4 exchange matrix.
        
        # Or we can compose:
        # Signal = f_csf * Ball + f_soma * Sphere + (1 - f_csf - f_soma) * Karger([Stick, Zeppelin])
        # This is hybrid.
        
        # The prompt asks for "SANDIX" -> SANDI with Exchange.
        # Full exchange SANDI would interpret all compartments as exchanging.
        # Let's try to model Neurite <-> Extra exchange, plus independent Soma and Ball, as this is physically most plausible
        # and easier to fit than fully coupled 4-compartment exchange.
        
        # Implementation:
        # Hybrid model function.
        
        print("Using Hybrid SANDIX: (Stick <-> Zeppelin) + Soma + CSF")
        
        # Ground Truth
        f_neurite = 0.4
        f_soma = 0.2
        f_csf = 0.1
        f_extra = 1.0 - f_neurite - f_soma - f_csf # 0.3
        
        # Fraction renormalization for the Karger block:
        # The Karger block represents fraction (f_neurite + f_extra).
        # Inside Karger: 
        #   f_stick_internal = f_neurite / (f_neurite + f_extra) = 0.4 / 0.7 ~= 0.57
        #   f_zep_internal = 1 - f_stick_internal
        
        f_total_tissue = f_neurite + f_extra
        f_stick_int = f_neurite / f_total_tissue
        
        tau_nex = 0.050 # 50ms exchange between Neurite and Extra
        
        diameter = 8e-6 # 8um Soma
        
        # Simulate
        key = jax.random.PRNGKey(42)
        
        # 1. Karger Part
        stick = C1Stick()
        zep = G2Zeppelin()
        karger = KargerExchange([stick, zep])
        
        # Karger Params: 
        # Stick: t, p, l_par
        # Zep: t, p, l_par, l_perp
        # Frac: f_stick_int
        # Exch: tau
        
        # Assume aligned
        theta, phi = 0.5, 0.5
        l_par = 2.5e-9
        l_perp = 0.5e-9 # Extracellular hindered
        
        k_params = jnp.array([
            theta, phi, l_par,
            theta, phi, l_par, l_perp,
            f_stick_int,
            tau_nex
        ])
        
        S_karger = karger.predict(k_params, acq)
        
        # 2. Soma Part
        sphere = SphereGPD()
        # Params: diameter, D
        D_soma = 2.0e-9 # Restricted cytosolic
        S_soma = sphere(
            bvals=bvals, gradient_directions=bvecs, 
            acquisition=acq, diameter=diameter, diffusion_constant=D_soma
        )
        
        # 3. CSF Part
        ball = G1Ball()
        D_csf = 3.0e-9
        S_csf = ball(bvals=bvals, lambda_iso=D_csf)
        
        # Combine
        S_total = f_total_tissue * S_karger + f_soma * S_soma + f_csf * S_csf
        
        # Add noise
        noise = jax.random.normal(key, S_total.shape) * 0.02
        data = jnp.abs(S_total + noise)[None, :] # 1 voxel for now? Let's replicate
        data = jnp.repeat(data, 20, axis=0)
        
        # Add variation to data
        data = jnp.abs(data + jax.random.normal(key, data.shape) * 0.01)
        
    
    # Define Fitting Model
    # Hybrid SANDIX Fitting Function
    
    stick = C1Stick()
    zep = G2Zeppelin()
    karger = KargerExchange([stick, zep])
    sphere = SphereGPD()
    ball = G1Ball()
    
    @jax.jit
    def hybrid_sandix_loss(params, signal_target, acq):
        # Params:
        # [0,1] theta, phi
        # [2] f_neurite
        # [3] f_soma
        # [4] f_csf
        # [5] tau_nex
        # [6] diameter
        # [7] l_perp_extra
        
        theta, phi = params[0], params[1]
        f_n = params[2]
        f_s = params[3]
        f_c = params[4]
        tau = params[5]
        diam = params[6]
        l_p_extra = params[7]
        
        l_par = 2.5e-9 # Fixed
        D_soma = 2.0e-9 # Fixed
        D_csf = 3.0e-9 # Fixed
        
        # Karger Calculation
        f_tissue = f_n + (1 - f_n - f_s - f_c) # = f_n + f_extra
        # Wait, f_extra is implicit.
        # f_n + f_s + f_c + f_e = 1
        # f_e = 1 - f_n - f_s - f_c
        
        f_tot = f_n + (1.0 - f_n - f_s - f_c)
        f_stick_rel = f_n / (f_tot + 1e-9)
        
        # Construct Karger Params
        kp = jnp.array([
            theta, phi, l_par,
            theta, phi, l_par, l_p_extra,
            f_stick_rel,
            tau
        ])
        
        S_k = karger.predict(kp, acq)
        
        S_s = sphere(
            bvals=acq.bvalues, gradient_directions=acq.gradient_directions,
            acquisition=acq, diameter=diam, diffusion_constant=D_soma
        )
        
        S_c = ball(bvals=acq.bvalues, lambda_iso=D_csf)
        
        # Combine
        # f_tot * S_k + f_s * S_s + f_c * S_c
        # Note: Optimization might behave weird if f_tot near 0
        pred = f_tot * S_k + f_s * S_s + f_c * S_c
        
        return jnp.mean((pred - signal_target)**2)

    optimizer = optax.adam(learning_rate=0.005)
    
    @jax.jit
    def fit_voxel(signal, init_p):
        opt_state = optimizer.init(init_p)
        
        def step(carry, i):
            p, state = carry
            l, grads = jax.value_and_grad(hybrid_sandix_loss)(p, signal, acq)
            updates, state = optimizer.update(grads, state, p)
            new_p = optax.apply_updates(p, updates)
            
            # Clip Constraints
            new_p = new_p.at[2].set(jnp.clip(new_p[2], 0.05, 0.8)) # f_n
            new_p = new_p.at[3].set(jnp.clip(new_p[3], 0.05, 0.8)) # f_s
            new_p = new_p.at[4].set(jnp.clip(new_p[4], 0.0, 0.5))  # f_c
            
            # Ensure sum <= 1
            sum_f = new_p[2] + new_p[3] + new_p[4]
            # scale if > 1? 
            # Simple approach: let them compete, but optimize well.
            
            new_p = new_p.at[5].set(jnp.clip(new_p[5], 0.001, 0.5)) # tau
            new_p = new_p.at[6].set(jnp.clip(new_p[6], 2e-6, 15e-6)) # diam
            new_p = new_p.at[7].set(jnp.clip(new_p[7], 0.0, 2.4e-9)) # l_perp
            
            return (new_p, state), l
            
        (final_p, _), _ = jax.lax.scan(step, (init_p, opt_state), jnp.arange(500))
        return final_p
        
    print("Fitting SANDIX...")
    # Init: t=0, p=0, fn=0.3, fs=0.3, fc=0.1, tau=0.1, d=8um, lp=0.5
    init_x = jnp.array([0., 0., 0.3, 0.3, 0.1, 0.1, 8e-6, 0.5e-9])
    
    fitted_x = jax.vmap(fit_voxel, in_axes=(0, None))(data, init_x)
    
    # Visualization
    f_n = fitted_x[:, 2]
    f_s = fitted_x[:, 3]
    tau = fitted_x[:, 5] * 1000 # ms
    diam = fitted_x[:, 6] * 1e6 # um
    
    print(f"Mean Neurite Frac: {jnp.mean(f_n):.2f}")
    print(f"Mean Soma Frac: {jnp.mean(f_s):.2f}")
    print(f"Mean Exchange Time: {jnp.mean(tau):.2f} ms")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 4, 1)
    plt.hist(f_n, bins=10, color='orange', edgecolor='black')
    plt.title('Neurite Fraction')
    
    plt.subplot(1, 4, 2)
    plt.hist(f_s, bins=10, color='cyan', edgecolor='black')
    plt.title('Soma Fraction')
    
    plt.subplot(1, 4, 3)
    plt.hist(tau, bins=10, color='purple', edgecolor='black')
    plt.title('Neurite-Extra Exchange (ms)')
    
    plt.subplot(1, 4, 4)
    plt.hist(diam, bins=10, color='green', edgecolor='black')
    plt.title('Soma Diameter (um)')
    
    plt.tight_layout()
    plt.savefig('sandix_results.png')
    print("Saved sandix_results.png")

if __name__ == "__main__":
    run_sandix_demo()
