
"""
Model Comparison Report using SDE-Generated Data.
Benchmarks:
1. DTI (Gaussian approximation) - Generic, ignores restriction.
2. Ball & Stick - Captures anisotropy, ignores finite diameter.
3. Restricted Cylinder - Captures finite diameter (High-G sensitivity).
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models, zeppelin, stick
from dmipy_jax.fitting.optimization import OptimistixFitter
from dmipy_jax.acquisition import JaxAcquisition
from pathlib import Path

def main():
    print("--- Connectome 2.0 Model Comparison Report ---")
    
    # 1. Load Data
    data_path = Path("results/connectome_sde_data.npz")
    if not data_path.exists():
        print("Data not found! Run demo_connectome_sde_simulation.py first.")
        return

    archive = np.load(data_path)
    signals = archive['signal'] # (N_meas,)
    bvals = archive['bvals']    # (N_meas,)
    gradients = archive['gradients'] # (N_meas, 3)
    
    # Create Scheme
    scheme = JaxAcquisition(
        bvalues=jnp.array(bvals),
        gradient_directions=jnp.array(gradients),
        delta=jnp.full(len(bvals), 0.012),
        Delta=jnp.full(len(bvals), 0.030)
    )
    
    print(f"Loaded {len(signals)} signals. Max b-value: {bvals.max():.2f}")
    
    # 2. Define Models and Helper
    
    # Helper to convert vector params to dict for JaxMultiCompartmentModel
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

    # Data is 1D (scalar signal for the ensemble of particles), so fit shape (1,1,1,N)
    data_reshaped = signals.reshape(1, 1, 1, -1)
    
    # Model A: Zeppelin
    print("\nFitting Model A: Zeppelin (Cylindrical Tensor)...")
    model_A = JaxMultiCompartmentModel(models=[zeppelin.Zeppelin()])
    
    ranges_A = [
        (0.0, np.pi), (0.0, 2*np.pi), # mu
        (0.0, 3e-9), (0.0, 3e-9)      # par, perp
    ]
    init_A = jnp.array([np.pi/2, np.pi, 2e-9, 0.5e-9])
    
    wrapper_A = make_vector_wrapper(model_A, model_A.parameter_names, model_A.parameter_cardinality)
    fitter_A = OptimistixFitter(wrapper_A, ranges_A)
    
    res_A, _ = fitter_A.fit(data_reshaped[0,0,0], scheme, init_A)
    pred_A = wrapper_A(res_A, scheme)
    mse_A = np.mean((signals - pred_A)**2)
    print(f"Model A MSE: {mse_A:.2e}")
    
    # Model B: Ball & Stick
    print("\nFitting Model B: Ball & Stick...")
    model_B = JaxMultiCompartmentModel(models=[stick.Stick(), gaussian_models.Ball()])
    
    ranges_B = [
        (0.0, np.pi), (0.0, 2*np.pi), (0.0, 3e-9), # Stick: mu, lambda
        (0.0, 3e-9),                               # Ball: lambda
        (0.0, 1.0)                                 # Mix
    ]
    init_B = jnp.array([np.pi/2, np.pi, 2e-9, 2e-9, 0.5])
    
    wrapper_B = make_vector_wrapper(model_B, model_B.parameter_names, model_B.parameter_cardinality)
    fitter_B = OptimistixFitter(wrapper_B, ranges_B)
    
    res_B, _ = fitter_B.fit(data_reshaped[0,0,0], scheme, init_B)
    pred_B = wrapper_B(res_B, scheme)
    mse_B = np.mean((signals - pred_B)**2)
    print(f"Model B MSE: {mse_B:.2e}")
    
    # Model C: Restricted Cylinder
    print("\nFitting Model C: Restricted Cylinder (Finite Radius)...")
    intra = cylinder_models.RestrictedCylinder(diameter=None) 
    extra = zeppelin.Zeppelin()
    iso = gaussian_models.Ball()
    
    model_C = JaxMultiCompartmentModel(models=[intra, extra, iso])
    
    ranges_C = [
        (0.0, np.pi), (0.0, 2*np.pi), (1e-7, 1e-5), (0.0, 3e-9), 
        (0.0, np.pi), (0.0, 2*np.pi), (0.0, 3e-9), (0.0, 3e-9),  
        (0.0, 3e-9),                                             
        (0.0, 1.0), (0.0, 1.0)                                   # Mix 1, Mix 2
    ]
    init_C = jnp.array([
        np.pi/2, np.pi, 10e-6, 1.7e-9, 
        np.pi/2, np.pi, 1.7e-9, 0.5e-9,
        3e-9,
        0.9, 0.05
    ])
    
    # Note: Model parameter order depends on how JaxMultiCompartmentModel stacks them.
    # Usually: [Model1, Model2, Model3, Mixes...]
    # But check actual order via model_C.parameter_names if fitting is weird.
    
    wrapper_C = make_vector_wrapper(model_C, model_C.parameter_names, model_C.parameter_cardinality)
    fitter_C = OptimistixFitter(wrapper_C, ranges_C)
    
    res_C, _ = fitter_C.fit(data_reshaped[0,0,0], scheme, init_C)
    pred_C = wrapper_C(res_C, scheme)
    mse_C = np.mean((signals - pred_C)**2)
    print(f"Model C MSE: {mse_C:.2e}")
    print(f"Model C Diam: {res_C[2]*1e6:.2f} um")
    
    # 3. Report
    print("\n--- Summary ---")
    print(f"Zeppelin (DTI-like): {mse_A:.2e}")
    print(f"Ball & Stick:        {mse_B:.2e}")
    print(f"Restricted Cylinder: {mse_C:.2e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(signals, 'k-', label='SDE Data', linewidth=2)
    plt.plot(pred_A, 'r--', label=f'Zeppelin (MSE={mse_A:.1e})')
    plt.plot(pred_B, 'g--', label=f'Ball&Stick (MSE={mse_B:.1e})')
    plt.plot(pred_C, 'b--', label=f'Restr. Cyl (MSE={mse_C:.1e})')
    plt.legend()
    plt.title("Model Comparison on Connectome SDE Data")
    plt.xlabel("Measurement Index")
    plt.ylabel("Signal Attenuation")
    plt.savefig("report_models.png")
    print("Saved report_models.png")

if __name__ == "__main__":
    main()
