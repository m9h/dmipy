import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.models.super_tissue_model import SuperTissueModel
from dmipy_jax.analysis.local_identifiability import check_model_identifiability

def run_identifiability_check():
    print("=== Checking SuperTissueModel Local Identifiability ===")
    
    # 1. Setup Model
    super_model = SuperTissueModel()
    
    # 2. Setup Acquisition (Standard Clinical Protocol)
    # b=1000, 2000, 3000 (30 dirs each) + b=0 (1 dir)
    # Total 91 measurements
    b0 = jnp.zeros(1)
    b1 = jnp.ones(30) * 1e9
    b2 = jnp.ones(30) * 2e9
    b3 = jnp.ones(30) * 3e9
    
    bvals = jnp.concatenate([b0, b1, b2, b3])
    
    # Random gradients on sphere
    key = jax.random.PRNGKey(42)
    bvecs = jax.random.normal(key, (len(bvals), 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    from collections import namedtuple
    Acq = namedtuple('Acq', ['bvalues', 'gradient_directions', 'delta', 'Delta'])
    acq_obj = Acq(bvals, bvecs, 0.01, 0.02) # SI units
    
    # 3. Setup Parameters (Random but Physical)
    # We avoid 0 to avoid singularities in Jacobian
    
    print("Parameter Dimensions:")
    print(super_model.parameter_cardinality)
    
    key, subkey = jax.random.split(key)
    rand_params = []
    param_names_flat = []
    
    for name in super_model.parameter_names:
        card = super_model.parameter_cardinality[name]
        ranges = super_model.parameter_ranges[name]
        
        # Flatten names
        if card == 1:
            param_names_flat.append(name)
            # Sample uniform
            val = jax.random.uniform(subkey, (1,), minval=ranges[0], maxval=ranges[1])
            rand_params.append(val)
        else:
            # Vector parameter: ranges is likely [(min, max), (min, max)...]
            # We iterate
            vec_parts = []
            for k in range(card):
                param_names_flat.append(f"{name}_{k}")
                dim_range = ranges[k]
                low = dim_range[0]
                high = dim_range[1]
                
                # Avoid singularities for angular parameters
                if "mu" in name and k == 0:
                     # Theta: avoid 0 and pi
                     low = max(low, 0.1)
                     high = min(high, jnp.pi - 0.1)
                
                val_k = jax.random.uniform(subkey, (1,), minval=low, maxval=high)
                vec_parts.append(val_k)
            
            rand_params.append(jnp.concatenate(vec_parts))
            
    full_params = jnp.concatenate(rand_params)
    
    print(f"Total Parameters: {len(full_params)}")
    
    # 3b. Sanity Check Forward Pass
    print("Running Forward Pass Sanity Check...")
    signal = super_model(full_params, acq_obj)
    if jnp.isnan(signal).any():
        print("!!! Forward Signal contains NaNs !!!")
        print("Signal:", signal)
        return
    print("Forward Pass Valid. Signal Mean:", jnp.mean(signal))

    # 4. Run Analysis
    result = check_model_identifiability(super_model, full_params, acq_obj, param_names_flat)
    
    # 5. Report
    print("\n--- Analysis Results ---")
    print(f"Total Parameters: {result.get('n_params', 'MISSING')}")
    print(f"Jacobian Rank: {result['rank']}")
    print(f"Identifiable: {result['is_identifiable']}")
    print(f"Condition Number: {result['condition_number']:.2e}")
    
    print("\nSingular Values:")
    print(result['singular_values'])
    
    if not result['is_identifiable']:
        print("\n--- UNIDENTIFIABLE PARAMETER SETS ---")
        for i, s in enumerate(result['collinear_sets']):
            print(f"\nSet {i+1} (Singular Value {s['singular_value']:.2e}):")
            # Sort by coefficient magnitude
            zipped = sorted(zip(s['params'], s['coefficients']), key=lambda x: abs(x[1]), reverse=True)
            for name, coef in zipped:
                print(f"  {coef:+.4f} * {name}")
                
if __name__ == "__main__":
    run_identifiability_check()
