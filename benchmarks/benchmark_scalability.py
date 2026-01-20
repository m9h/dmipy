import time
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.signal_models.sphere_models import Sphere
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def run_scalability_benchmark():
    print("=== Dmipy-JAX Scalability Benchmark ===")
    
    # 1. Setup Model (NODDI-like: Cylinder + Sphere + Ball)
    # Using Cylinder and Sphere for now as Ball might be standard Gaussian
    cylinder = RestrictedCylinder(lambda_par=1.7e-9, diameter=6e-6)
    sphere = Sphere(diameter=8e-6)
    
    model = JaxMultiCompartmentModel([cylinder, sphere])
    
    # 2. Setup Acquisition
    N_dirs = 64
    bval = 3000e6
    bvecs = np.random.randn(N_dirs, 3)
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvals = np.ones(N_dirs) * bval
    # Incorporate b0
    bvecs = np.vstack([np.zeros((1, 3)), bvecs])
    bvals = np.hstack([0, bvals])
    
    acq = JaxAcquisition(bvalues=bvals, bvectors=bvecs, small_delta=0.02, big_delta=0.04)
    
    # 3. Define dataset sizes to test
    sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000]
    
    # Generate Synthetic Data (One voxel replicated)
    # True parameters
    # Cylinder: mu (orientation) needs to be fitted? Or fixed? 
    # Usually in fit() we fit intrinsic params + orientation.
    # For this benchmark, let's fix orientation to keep it simple or allow fit?
    # To test memory, we need full fit.
    
    # Let's generate data dependent on params
    # True params:
    # Cylinder: mu=[1,0,0]
    # Fractions: 0.5, 0.5
    
    # We need to construct a parameter dictionary for simulation
    # The JaxMultiCompartmentModel handles parameters.
    # parameter_names will look like: ['mu', 'partial_volume_0', 'partial_volume_1']
    # Wait, 'mu' comes from RestrictedCylinder. Sphere has no orientation? 
    # Sphere usually has no orientation.
    
    # Let's check model.parameter_names
    print(f"Model parameters: {model.parameter_names}")
    
    # Construct "true" params for simulation
    true_params = {}
    if 'mu' in model.parameter_names:
        true_params['mu'] = jnp.array([1., 0., 0.])
    
    # partial volumes
    true_params['partial_volume_0'] = 0.6
    true_params['partial_volume_1'] = 0.4
    
    # Simulate signal for 1 voxel
    # signal_1vox: (N_meas,)
    signal_1vox = model(true_params, acq)
    
    print(f"\nTesting sizes: {sizes}")
    
    for N in sizes:
        print(f"\n--- Testing N = {N} voxels ---")
        
        # Replicate data
        data = jnp.tile(signal_1vox, (N, 1))
        # Add some noise to make fit non-trivial?
        # data = data + 0.01 * jax.random.normal(jax.random.PRNGKey(0), data.shape)
        
        # Check memory usage before fit (approximate if possible, or just run)
        
        start_time = time.time()
        try:
            # We need to catch OOM
            # JAX operations are async, so we need to block
            print("Starting fit...")
            
            # Use batch_size argument if implemented, otherwise default fit
            # Check if fit accepts batch_size
            import inspect
            sig = inspect.signature(model.fit)
            has_batch = 'batch_size' in sig.parameters
            
            if has_batch:
                print("Using batch_size=100_000")
                fitted = model.fit(acq, data, batch_size=100_000)
            else:
                print("Using default fit (no batching)")
                fitted = model.fit(acq, data)
            
            # Force synchronization
            # Access one element to ensure computation is done
            if isinstance(fitted, dict):
                 # Block on one value
                 _ = fitted[list(fitted.keys())[0]].block_until_ready()
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = N / duration
            
            print(f"Success!")
            print(f"Time: {duration:.2f} s")
            print(f"Throughput: {throughput:.2e} voxels/s")
            
        except Exception as e:
            print(f"FAILED with error: {e}")
            # If it's an OOM, JAX might print it to stderr or raise XlaRuntimeError
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_scalability_benchmark()
