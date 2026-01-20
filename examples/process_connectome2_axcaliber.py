
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path
from dmipy_jax.io.connectome2 import fetch_connectome2, load_connectome2_mri
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.distributions.distributions import DD1Gamma
from dmipy_jax.distributions.distribute_models import DistributedModel
from dmipy_jax.acquisition import JaxAcquisition
import optax
import equinox as eqx

def main():
    print("=== Connectome 2.0 AxCaliber Analysis ===")
    
    # 1. Load Data
    # Assuming fetch works or data is manual.
    try:
        data_dict = load_connectome2_mri()
        data = data_dict['dwi']
        bvals = data_dict['bvals']
        bvecs = data_dict['bvecs']
        print(f"Data Loaded: {data.shape}")
    except Exception as e:
        print(f"Data load failed ({e}). using MOCK data for demonstration.")
        # Mocking behavior similar to Connectome 2.0
        # D1 = 13ms, D2 = 30ms
        N = 100
        data = jnp.ones((10, 10, 10, N))
        bvals = jnp.concatenate([jnp.linspace(50, 6000, N//2), jnp.linspace(200, 17800, N//2)])
        bvecs = jnp.zeros((N, 3)); bvecs = bvecs.at[:,0].set(1.0)
    
    # 2. Assign Diffusion Times
    # Convert bvals to SI (s/m^2) if they are in s/mm^2 (standard for bvals)
    if jnp.max(bvals) < 1e8:
        print("Converting bvalues to SI (s/m^2)...")
        bvals = bvals * 1e6

    # Heuristic based on description:
    # D=13ms (small Delta 13?) -> Delta=0.013? Or is it Diffusion Time tau?
    # Usually "Delta" in Connectome scanner context refers to big Delta.
    # Protocol: 
    #   Set A: Delta = 13 ms? Wait, gradients need time. 
    #   Maybe Delta=13ms is impossible for b=6000 unless delta is very close.
    #   Actully, "Diffusion time 13ms" might mean Delta - delta/3 = 13ms.
    #   Let's assume standard PGSE.
    #   Let's assign Delta = 0.020 for set A, 0.040 for set B?
    #   Wait, the description said "D=13 ms and 30 ms".
    #   I'll assume distinct b-value ranges or interleaved?
    #   If bvals > 10000, it MUST be D=30ms (to achieve high b).
    #   If bvals < 6000, it could be D=13ms.
    
    # Simple split for demo:
    # If we had real data, we'd check json.
    # Here, let's create synthetic scheme with explicit Deltas.
    
    # Assume 2 shells for now:
    big_delta = jnp.where(bvals > 7000e6, 0.030, 0.013) 
    
    # 6000 s/mm2 = 6000 * 1e6 s/m2 = 6e9.
    # Safe logic:
    big_delta = jnp.where(bvals > 7e9, 0.030, 0.013)
    small_delta = jnp.ones_like(bvals) * 0.010 # Assumed
    
    # Use correct field names for JaxAcquisition: Delta (big), delta (small)
    scheme = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs, 
                            Delta=big_delta, delta=small_delta)
    
    print(f"Scheme created. Delta values: {jnp.unique(scheme.Delta)}")
    
    
    # 3. Define AxCaliber Model
    cylinder = RestrictedCylinder()
    gamma_dist = DD1Gamma()
    
    # Distribute 'diameter' using Gamma
    # DD1Gamma integrates over 'x' which maps to 'diameter' here.
    axcaliber = DistributedModel(
        cylinder, gamma_dist,
        target_parameter='diameter'
    )
    
    print("Model initialized: AxCaliber (Gamma-Distributed Cylinder)")
    
    # 4. Generate Synthetic Data (Forward Pass)
    print("Generating synthetic data...")
    # Ground Truth Parameters
    gt_params = {
        'lambda_par': 1.7e-9,  # Intracellular diffusivity
        'mu': jnp.array([1.57, 0.0]), # x-axis
        'alpha': 10.0,
        'beta': 0.5e-6 # 0.5 um -> mean = 5 um
    }
    
    # Note: Must pass acquisition timings (big_delta, small_delta) to the model kwargs
    acq_kwargs = {
        'big_delta': scheme.Delta,
        'small_delta': scheme.delta
    }
    
    signal_gt = axcaliber(bvals, bvecs, **gt_params, **acq_kwargs)
    
    # Add noise?
    signal_noisy = signal_gt # No noise for clean demo first
    
    print(f"Signal generated. Mean: {jnp.mean(signal_noisy):.4f}")
    
    # 5. Optimization
    print("Starting optimization...")
    
    # Initial Guess
    init_params = {
        'lambda_par': 1.5e-9,
        'mu': jnp.array([1.5, 0.1]),
        'alpha': 5.0,
        'beta': 1.0e-6
    }
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(init_params)
    
    @jax.jit
    def loss_fn(params, data):
        # Constraints (Softplus)
        p_constrained = {
            'lambda_par': jax.nn.softplus(params['lambda_par']),
            'mu': params['mu'], # angles unconstrained
            'alpha': jax.nn.softplus(params['alpha']),
            'beta': jax.nn.softplus(params['beta'])
        }
        # Pass acquisition kwargs here too!
        pred = axcaliber(bvals, bvecs, **p_constrained, **acq_kwargs)
        return jnp.mean((pred - data)**2)
    
    @jax.jit
    def step(params, opt_state, data):
        grads = jax.grad(loss_fn)(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_fn(params, data)
        
    # Transformation (Inverse Softplus for init)
    def inv_softplus(x): return jnp.log(jnp.exp(x) - 1.0)
    
    params_optim = {
        'lambda_par': inv_softplus(init_params['lambda_par']),
        'mu': init_params['mu'],
        'alpha': inv_softplus(init_params['alpha']),
        'beta': inv_softplus(init_params['beta'])
    }
    
    start_time = time.time()
    for i in range(500):
        params_optim, opt_state, loss = step(params_optim, opt_state, signal_noisy)
        if i % 100 == 0:
            print(f"Iter {i}: Loss {loss:.6f}")
            
    print(f"Fitting complete in {time.time()-start_time:.2f}s")
    
    # Recovered
    final_params = {
        'lambda_par': jax.nn.softplus(params_optim['lambda_par']),
        'mu': params_optim['mu'],
        'alpha': jax.nn.softplus(params_optim['alpha']),
        'beta': jax.nn.softplus(params_optim['beta'])
    }
    
    print("=== Results ===")
    print(f"GT Alpha: {gt_params['alpha']:.4f} | Rec: {final_params['alpha']:.4f}")
    print(f"GT Beta:  {gt_params['beta']*1e6:.4f} um | Rec: {final_params['beta']*1e6:.4f} um")
    
    # Mean Diameter = alpha * beta
    mean_gt = gt_params['alpha'] * gt_params['beta'] * 1e6
    mean_rec = final_params['alpha'] * final_params['beta'] * 1e6
    print(f"GT Mean Diameter: {mean_gt:.4f} um | Rec: {mean_rec:.4f} um")
    
if __name__ == "__main__":
    main()
