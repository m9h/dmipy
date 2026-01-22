
"""
Debug script for Restricted Cylinder fit failure.
Objective: Compare OptimistixFitter vs VoxelFitter and inspect fitted parameters.
"""

import numpy as np
import jax.numpy as jnp
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models, zeppelin
from dmipy_jax.fitting.optimization import OptimistixFitter, VoxelFitter
from dmipy_jax.acquisition import JaxAcquisition
from pathlib import Path

def main():
    print("--- Debugging Restricted Cylinder Fit ---")
    
    # 1. Load Data
    data_path = Path("results/connectome_sde_data.npz")
    if not data_path.exists():
        print("Data missing.")
        return

    archive = np.load(data_path)
    signals = archive['signal']
    bvals = archive['bvals']
    gradients = archive['gradients']
    
    scheme = JaxAcquisition(
        bvalues=jnp.array(bvals),
        gradient_directions=jnp.array(gradients),
        delta=jnp.full(len(bvals), 0.012),
        Delta=jnp.full(len(bvals), 0.030)
    )
    
    # 2. Define Model
    print("Defining Model C: RestrictedCylinder + Zeppelin + Ball...")
    intra = cylinder_models.RestrictedCylinder(diameter=None) 
    extra = zeppelin.Zeppelin()
    iso = gaussian_models.Ball()
    
    model_C = JaxMultiCompartmentModel(models=[intra, extra, iso])
    
    ranges_C = [
        (0.0, np.pi), (0.0, 2*np.pi), (1e-7, 20e-6), (0.1e-9, 3e-9), # Cyl: mu, diam, diff
        (0.0, np.pi), (0.0, 2*np.pi), (0.1e-9, 3e-9), (0.1e-9, 3e-9),# Zep: mu, par, perp
        (0.1e-9, 3e-9),                                              # Ball
        (0.0, 1.0), (0.0, 1.0)                                       # Mix 1, Mix 2
    ]
    
    # Init from previous script
    init_C = jnp.array([
        np.pi/2, np.pi, 5e-6, 1.7e-9, 
        np.pi/2, np.pi, 1.7e-9, 0.5e-9,
        3e-9,
        0.3, 0.3
    ])

    # Helper wrapper (same as before)
    def make_vector_wrapper(model, param_names, cardinalities):
        def wrapper(params_vec, acquisition):
            idx = 0
            params_dict = {}
            for name in param_names:
                card = cardinalities.get(name, 1)
                if card == 1:
                    val = params_vec[idx]
                    idx += 1
                else:
                    val = params_vec[idx:idx+card]
                    idx += card
                params_dict[name] = val
            return model(params_dict, acquisition)
        return wrapper

    wrapper_C = make_vector_wrapper(model_C, model_C.parameter_names, model_C.parameter_cardinality)
    
    # Test 1: Optimistix (Results from report) with Multi-Start
    print("\n--- Test 1: OptimistixFitter (Unconstrained LM) with Multi-Start ---")
    
    # Grid search for Diameter and Stick/Cyl Volume Fraction
    # Simulation is Macroscopic -> Free Diffusion -> Large Diameter limit.
    diams = [10.0e-6, 30.0e-6, 60.0e-6] 
    mixes = [0.1, 0.5, 0.9]
    
    best_mse = np.inf
    best_res = None
    
    for d in diams:
        for m in mixes:
            print(f"Trying Init: Diam={d*1e6:.1f}um, Mix={m}")
            # Update init_C
            # indices: 2 (diam), 9 (mix1), 10 (mix2 - extra/iso split?)
            # Our mix params: mix1 (intra?), mix2 (extra?) based on model list order [intra, extra, iso]
            # Params: intra..., extra..., iso..., mix_intra_extra, mix_remainder_iso?
            # JaxMultiCompartmentModel mixing params are N-1 usually?
            # Or [vol1, vol2, vol3]? No, usually N-1 fractions.
            # But earlier ranges had 2 mix params for 3 models. (v1, v2). v3 = 1-v1-v2.
            
            # Construct init
            curr_init = jnp.array(init_C)
            curr_init = curr_init.at[2].set(d)
            curr_init = curr_init.at[9].set(m) # Intra volume
            curr_init = curr_init.at[10].set((1-m)/2) # Extra volume
            
            fitter = OptimistixFitter(wrapper_C, ranges_C)
            res, _ = fitter.fit(signals, scheme, curr_init)
            
            # Eval
            pred = wrapper_C(res, scheme)
            mse = np.mean((signals - pred)**2)
            print(f"  -> MSE: {mse:.2e}, Diam Fit: {res[2]*1e6:.2f}um")
            
            if mse < best_mse:
                best_mse = mse
                best_res = res
                
    print(f"\nBest MSE: {best_mse:.2e}")
    print(f"Best Diameter: {best_res[2]*1e6:.2f} um")
    
    # Check if this fixes the RMSE issue (Target 2e-3 range like BallStick?)
    # BallStick was 2.5e-3. 
    # If we get < 0.01, that is success.
    
    if best_mse < 0.05:
        print("SUCCESS: Multi-start fixed the convergence.")
    else:
        print("FAILURE: Still poor fit. Model mismatch likely.")
if __name__ == "__main__":
    main()
