import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import Stick, Ball
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.fitting.training import StochasticTrainer

def main():
    print("--- Demo: Stochastic Training with Optax ---")

    # 1. Setup Acquisition
    print("Setting up acquisition...")
    bvalues = jnp.array([1000.0] * 30 + [2000.0] * 30 + [3000.0] * 30)
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    vecs = jax.random.normal(k1, (90, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    acq = acquisition_scheme_from_bvalues(bvalues, vecs)
    
    # 2. Setup Model (Ball & Stick)
    print("Setting up Stick + Ball model...")
    stick = Stick()
    ball = Ball()
    model = JaxMultiCompartmentModel([stick, ball])
    
    print("Parameter names:", model.parameter_names)

    # 3. Simulate Ground Truth Data
    final_dict = {}
    for name in model.parameter_names:
        if 'mu' in name:
            final_dict[name] = jnp.array([1.0, 0.5]) # theta=1.0, phi=0.5
        elif 'lambda_par' in name:
            final_dict[name] = jnp.array(1.7e-9)
        elif 'lambda_iso' in name:
            final_dict[name] = jnp.array(0.8e-9) 
        elif 'diffusivity' in name:
            final_dict[name] = jnp.array(0.8e-9)
        elif 'partial_volume_0' in name:
             final_dict[name] = jnp.array(0.6)
        elif 'partial_volume_1' in name:
             final_dict[name] = jnp.array(0.4)
            
    print("GT Dictionary:", final_dict)
    
    gt_flat = model.parameter_dictionary_to_array(final_dict)
    
    # Simulate
    signal_clean = model(final_dict, acq)
    
    # Add Noise (Rician)
    sigma = 0.02 # SNR = 50
    noise_r = jax.random.normal(k2, signal_clean.shape) * sigma
    noise_i = jax.random.normal(k3, signal_clean.shape) * sigma
    signal_noisy = jnp.sqrt((signal_clean + noise_r)**2 + noise_i**2)
    
    print(f"Data simulated. SNR ~ {1/sigma:.1f}.")

    # 4. Determine Scales
    print("Calculating scales...")
    scales_list = []
<<<<<<< HEAD
    # Replicate logic roughly from JaxMultiCompartmentModel.fit or just manual since we know the model
    # Stick: mu (2), lambda_par (1)
    # Ball: lambda_iso (1)
    # Fractions: (2)
    
    # Just iterate parameter names and use ranges
    # WARNING: This relies on specific ordering in flat array matching parameter_names iteration
    # which JaxMultiCompartmentModel enforces.
=======
>>>>>>> recovery_work_v2
    
    for name in model.parameter_names:
        card = model.parameter_cardinality[name]
        rng = model.parameter_ranges[name]
        
        current_scales = []
<<<<<<< HEAD
        
        # Helper to pick scale from range (min, max)
        def get_scale(r):
            if isinstance(r, (list, tuple)):
                 l, h = r
                 # If h is finite and != 0, use it. Else 1.0.
                 # Actually ranges can be list of tuples for vector params
                 # Simpler: if h < 1e-6, use 1e-9? 
                 if abs(h) < 1e-6 and h != 0: return h
                 if abs(h) > 1e6: return h
                 return 1.0 # Angles and Fractions (~1)
=======
        def get_scale(r):
            if isinstance(r, (list, tuple)):
                 l, h = r
                 if abs(h) < 1e-6 and h != 0: return h
                 if abs(h) > 1e6: return h
                 return 1.0 
>>>>>>> recovery_work_v2
            return 1.0
            
        if card == 1:
            s = get_scale(rng)
            current_scales.append(s)
        else:
<<<<<<< HEAD
            # Vector
            if isinstance(rng, tuple) and len(rng)==2 and isinstance(rng[0], (int, float)):
                 # Uniform range
                 s = get_scale(rng)
                 current_scales.extend([s]*card)
            else:
                 # List of ranges
=======
            if isinstance(rng, tuple) and len(rng)==2 and isinstance(rng[0], (int, float)):
                 s = get_scale(rng)
                 current_scales.extend([s]*card)
            else:
>>>>>>> recovery_work_v2
                 for r in rng:
                     current_scales.append(get_scale(r))
                     
        scales_list.extend(current_scales)
        
    scales = jnp.array(scales_list)
    print("Scales:", scales)

    # 5. Setup Stochastic Trainer
    print("Initializing StochasticTrainer...")
    
<<<<<<< HEAD
    # Optimizer
    # With scaling, params are ~1. Learing rate 1e-2 should be fine.
    lr = 5e-2 
    # Use scheduler to decay
=======
    # Constrained optimization requires slower learning rate usually?
    lr = 5e-2
>>>>>>> recovery_work_v2
    scheduler = optax.exponential_decay(init_value=lr, transition_steps=500, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=scheduler)
    
    trainer = StochasticTrainer(model, optimizer)
    
    # Initial Guess
    # Normalize GT by scales to perturb in "internal" space, then rescale
    gt_internal = gt_flat / scales
<<<<<<< HEAD
    perturbation = jax.random.normal(k3, gt_internal.shape) * 0.2 # 20% perturbation
    init_internal = gt_internal + perturbation
    
    # Clamp fractions to 0-1 range roughly if they went wild, though trainer doesn't enforce
    # Just let it be.
    init_flat = init_internal * scales
    
    # 6. Fit
    print("Starting fit (Rician Loss)...")
    fitted_flat = trainer.fit(
        init_flat, 
        acq, 
        signal_noisy, 
        epochs=2000, 
        loss_type='rician',
        sigma=sigma,
        scales=scales,
        verbose=True
    )
    
=======
    perturbation = jax.random.normal(k3, gt_internal.shape) * 0.1 # 10% perturbation
    init_internal = gt_internal + perturbation
    
    init_flat = init_internal * scales
    
    # 6. Fit
    print("Starting fit (Rician Loss) with Constraints...")
    
    # Define unwrap_fn to enforce constraints
    # params_flat comes in with scaling APPLIED.
    # We want to map them to physical range.
    
    def unwrap_fn(p_flat):
        # We assume p_flat structure matches model.parameter_names order
        # mu: unconstrained (angles)
        # lambda_par: softplus (must be positive)
        # lambda_iso: softplus (must be positive)
        # partial_volumes: softplus (must be positive)
        
        p_constrained = []
        idx = 0
        for name in model.parameter_names:
            card = model.parameter_cardinality[name]
            p_sub = p_flat[idx:idx+card]
            
            if 'mu' in name:
                p_c = p_sub 
            else:
                 # Diffusivities and fractions -> abs (simple positivity)
                 # We avoid softplus because p_sub is already scaled (e.g. 1e-9)
                 # softplus(1e-9) ~ log(2) ~ 0.69 (HUGE error)
                 p_c = jnp.abs(p_sub)
                 
            p_constrained.append(p_c)
            idx += card
            
        return [jnp.concatenate(p_constrained)]

    fitted_flat_raw = trainer.fit(
        init_flat, 
        acq, 
        signal_noisy, 
        epochs=3000, 
        loss_type='rician',
        sigma=sigma,
        scales=scales,
        unwrap_fn=unwrap_fn,
        verbose=True
    )
    
    # Apply unwrap_fn to final result to get physical values
    fitted_flat = unwrap_fn(fitted_flat_raw)[0]
    
>>>>>>> recovery_work_v2
    # 7. Results
    fitted_dict = model.parameter_array_to_dictionary(fitted_flat)
    
    print("\n--- Results ---")
    print(f"{'Parameter':<25} {'Ground Truth':<25} {'Fitted':<25} {'Error %':<10}")
    
    for name in model.parameter_names:
        gt_val = final_dict[name]
        fit_val = fitted_dict[name]
        
        # Handle scalar vs vector
        if jnp.size(gt_val) > 1:
             val_str_gt = str(jnp.round(gt_val, 3))
             val_str_fit = str(jnp.round(fit_val, 3))
             err = jnp.linalg.norm(gt_val - fit_val) / (jnp.linalg.norm(gt_val) + 1e-12) * 100
        else:
             val_str_gt = f"{gt_val:.3e}"
             val_str_fit = f"{fit_val:.3e}"
             err = jnp.abs(gt_val - fit_val) / (jnp.abs(gt_val) + 1e-12) * 100
             
        print(f"{name:<25} {val_str_gt:<25} {val_str_fit:<25} {err:.2f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()
