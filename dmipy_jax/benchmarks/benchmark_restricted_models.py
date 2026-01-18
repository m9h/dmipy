
import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models import cylinder_models, sphere_models, plane_models

def run_benchmark():
    print("# Dmipy-JAX Restricted Models Benchmark")
    print(f"Device: {jax.devices()[0]}")
    print("| Model | JIT Time (s) | 100k Voxels Time (ms) | Throughput (Voxels/s) |")
    print("|---|---|---|---|")

    models_to_test = [
        ("Restricted Cylinder (Soderman)", cylinder_models.RestrictedCylinder(), 
         {'diameter': 5e-6, 'big_delta': 0.03, 'small_delta': 0.01}),
        
        ("Restricted Cylinder (Callaghan, N=20)", cylinder_models.CallaghanRestrictedCylinder(number_of_roots=20, number_of_functions=20), 
         {'diameter': 5e-6, 'tau': 0.02, 'diffusion_perpendicular': 1e-9}),
         
        ("Sphere (Stejskal-Tanner)", sphere_models.SphereStejskalTanner(), 
         {'diameter': 5e-6, 'big_delta': 0.03, 'small_delta': 0.01}),
         
        ("Sphere (Callaghan, N=20)", sphere_models.SphereCallaghan(number_of_roots=20, number_of_functions=20), 
         {'diameter': 5e-6, 'diffusion_constant': 1e-9, 'big_delta': 0.03, 'small_delta': 0.01}),
         
        ("Plane (Stejskal-Tanner)", plane_models.PlaneStejskalTanner(), 
         {'diameter': 5e-6, 'big_delta': 0.03, 'small_delta': 0.01}),

        ("Plane (Callaghan, N=20)", plane_models.PlaneCallaghan(number_of_roots=20), 
         {'diameter': 5e-6, 'diffusion_constant': 1e-9, 'big_delta': 0.03, 'small_delta': 0.01}),
    ]

    N_voxels = 100_000
    # Simulate a realistic acquisition: 60 shells/bvecs
    N_bvecs = 60
    
    # We benchmark the kernel for a single voxel (vectorized over bvecs) usually,
    # OR we benchmark vmapping over voxels.
    # Typical use case: Fitting 1 million voxels.
    # In dmipy_jax, we often fit voxel-wise or chunked.
    # Let's benchmark a large batch call: (N_voxels, N_bvecs)
    # Most models currently operate on (bvals, bvecs) for a SINGLE voxel parameters (scalars)
    # OR they broadcast.
    # To benchmark throughput, we should vmap over parameters.
    
    # Generate common acquisition
    bvals = jnp.ones(N_bvecs) * 3000.0
    bvecs = jnp.zeros((N_bvecs, 3)); bvecs = bvecs.at[:, 0].set(1.0) # all aligned
    mu = jnp.array([1.0, 0.0, 0.0]) # Fiber direction
    
    for name, model, params in models_to_test:
        # Prepare parameters as arrays of size N_voxels
        # Actually our models take scalar parameters usually?
        # Let's checking if we need to vmap the model.
        # Yes, standard models take scalar params. We need to vmap to handle N_voxels.
        
        # Prepare inputs
        # We want to measure the throughput of calculating signal for N_voxels different parameters.
        
        # Create a vmapped execution function
        # Signature: predict(params_dict) -> signal (N_bvals,)
        
        # Helper wrapper to unpack dictionary
        def predict_single(p):
            # p is dict of scalars
            return model(bvals, bvecs, **p, mu=mu, lambda_par=1.7e-9)
            
        predict_batch = jax.jit(jax.vmap(predict_single))
        
        # Create random parameters bundle
        batch_params = {}
        key = jax.random.PRNGKey(0)
        for k, v in params.items():
            if isinstance(v, float) or isinstance(v, int):
                # Add some noise to prevent constant folding optimizations over batch?
                # Though JAX usually doesn't fold valid math.
                # Generate random array
                batch_params[k] = jnp.full((N_voxels,), v) + jax.random.uniform(key, (N_voxels,), minval=-0.1*v, maxval=0.1*v)
                key, _ = jax.random.split(key)
        
        # 1. JIT Warmup
        try:
            start_jit = time.time()
            _ = predict_batch(batch_params).block_until_ready()
            jit_time = time.time() - start_jit
        except Exception as e:
            print(f"| {name} | FAILED | {e} | - |")
            continue
            
        # 2. Execution
        start_exec = time.time()
        _ = predict_batch(batch_params).block_until_ready()
        end_exec = time.time()
        
        exec_time = end_exec - start_exec
        exec_time_ms = exec_time * 1000
        throughput = N_voxels / exec_time
        
        print(f"| {name} | {jit_time:.4f} | {exec_time_ms:.2f} | {throughput:.2e} |")

if __name__ == "__main__":
    run_benchmark()
