import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dmipy_jax.bayesian.discovery import BayesianDiscovery
from dmipy_jax.models.super_tissue_model import SuperTissueModel
# We need an acquisition scheme
# Let's create a simple one manually or use a helper if available.
# We'll just pass arrays since our simple acquisition wrapper supports it.

def test_bayesian_discovery_simple():
    # 1. Setup Ground Truth
    # Stick (f=0.6) + Ball (f=0.4)
    # Others = 0
    
    # Create reduced model for verification stability
    from dmipy_jax.signal_models.stick import Stick
    from dmipy_jax.signal_models.gaussian_models import Ball
    
    super_model = SuperTissueModel(models=[Stick(), Ball()])
    
    # Create an acquisition
    # 3 shells: b=1000, 2000, 3000. 10 dirs each. + b=0
    bvals = jnp.concatenate([jnp.zeros(1), jnp.ones(10)*1e9, jnp.ones(10)*2e9, jnp.ones(10)*3e9])
    
    # Random gradients
    key = jax.random.PRNGKey(0)
    bvecs = jax.random.normal(key, (31, 3))
    bvecs = bvecs / jnp.linalg.norm(bvecs, axis=1, keepdims=True)
    
    acquisition_kwargs = {'delta': 0.01, 'Delta': 0.02} # SI units
    
    # Construct params for GT (Stick + Ball)
    
    # 0: Stick
    mu_stick = jnp.array([1.57, 0.0])
    lambda_par_stick = jnp.array([1.7e-9])
    
    # 1: Ball
    lambda_iso_ball = jnp.array([3e-9])
    
    # Fractions (2 models)
    f_stick = 0.6
    f_ball = 0.4
    
    param_vector = []
    param_vector.append(mu_stick)
    param_vector.append(lambda_par_stick)
    param_vector.append(lambda_iso_ball)
    param_vector.append(jnp.array([f_stick]))
    param_vector.append(jnp.array([f_ball]))
    
    gt_params = jnp.concatenate([p.flatten() for p in param_vector])
    
    # Predict GT Signal
    from collections import namedtuple
    Acq = namedtuple('Acq', ['bvalues', 'gradient_directions', 'delta', 'Delta'])
    acq_obj = Acq(bvals, bvecs, 0.01, 0.02)
    
    gt_signal = super_model(gt_params, acq_obj)
    
    # Sanity Check
    print("GT Signal:", gt_signal[:5])
    if jnp.any(jnp.isnan(gt_signal)):
        raise ValueError("GT Signal contains NaNs!")
        
    print("Sanity Check Passed.")
    
    # Add noise
    sigma_noise = 0.02
    noisy_signal = gt_signal + jax.random.normal(key, gt_signal.shape) * sigma_noise
    
    # Create init_params dict (generic)
    init_params = {}
    current_idx = 0
    noisy_start_params = gt_params + jax.random.normal(key, gt_params.shape) * 0.1 * gt_params
    
    for name in super_model.parameter_names:
        card = super_model.parameter_cardinality[name]
        if card == 1:
            init_params[name] = noisy_start_params[current_idx]
            current_idx += 1
        else:
            for k in range(card):
                init_params[f"{name}_{k}"] = noisy_start_params[current_idx]
                current_idx += 1
                
    # 2. Run Discovery with reduced model
    discovery = BayesianDiscovery(acquisition_model=super_model)
    
    result = discovery.discover_microstructure(
        noisy_signal, bvals, bvecs, acquisition_kwargs,
        epsilon=0.05,
        init_params=init_params,
        num_samples=100,
        num_warmup=100
    )
    
    probs = result['probabilities']
    print("Marginal Probabilities:", probs)
    print("Mean Weights:", result['mean_weights'])
    
    # 3. Assertions
    # Stick (0) and Ball (1) should be high
    assert probs[0] > 0.8, f"Stick detection failed: {probs[0]}"
    assert probs[1] > 0.8, f"Ball detection failed: {probs[1]}"

if __name__ == "__main__":
    test_bayesian_discovery_simple()
