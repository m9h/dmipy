
import time
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.cylinder import C1Stick
from dmipy_jax.acquisition import JaxAcquisition

def main():
    print("Setting up benchmark...")
    
    # 1. Setup Parameters
    N_VOXELS = 50000
    N_GRADIENTS = 60
    
    # 2. Data Generation
    # Mock Acquisition
    # Shell at b=1000 with 60 gradient directions
    bvals = jnp.ones(N_GRADIENTS) * 1000.0
    
    # Random gradients on sphere
    grads_np = np.random.randn(N_GRADIENTS, 3)
    grads_np /= np.linalg.norm(grads_np, axis=1, keepdims=True)
    grads = jnp.array(grads_np)
    
    # Create JaxAcquisition object
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=grads)
    
    # Random Model Parameters for 50,000 voxels
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # mu: (N, 2) angles [theta, phi]
    mu = jax.random.uniform(k1, shape=(N_VOXELS, 2), minval=0.0, maxval=np.pi)
    
    # lambda_par: (N,) diffusivity [0.1e-9, 3.0e-9]
    lambda_par = jax.random.uniform(k2, shape=(N_VOXELS,), minval=0.1e-9, maxval=3.0e-9)
    
    # Instantiate Model
    # Note: Import as C1Stick, alias as Stick model per request context mostly referring to Stick.
    # Using 'Stick' implies the C1Stick class.
    model = C1Stick()

    # Define prediction wrapper to handle explicit parameter passing
    # The model.__call__ expects (bvals, grads, **kwargs)
    def predict_one(mu_val, lambda_val):
        return model(acq.bvalues, acq.gradient_directions, mu=mu_val, lambda_par=lambda_val)

    # 3. Baseline: Serial Loop
    def run_serial_loop():
        print(f"Running Serial Loop on {N_VOXELS} voxels...")
        # We invoke the prediction one by one.
        # To strictly measure execution time including dispatch overhead (which is the killer in Python loops),
        # we iterate and block.
        
        start_time = time.perf_counter()
        for i in range(N_VOXELS):
            if i % 5000 == 0:
                print(f"Processing {i}/{N_VOXELS}", end='\r')
            # Slicing JAX arrays to get scalar-like JAX arrays
            mu_i = mu[i]
            lambda_i = lambda_par[i]
            
            res = predict_one(mu_i, lambda_i)
            # Ensure computation is done
            res.block_until_ready()
            
        end_time = time.perf_counter()
        return end_time - start_time

    # 4. Challenger: JAX Vmap
    # Create vmapped function
    # Maps over axis 0 of mu and lambda_par
    vmapped_predict = jax.vmap(predict_one, in_axes=(0, 0))
    
    # JIT compile the vmapped function for maximum performance
    # The instructions mention "trigger JAX JIT compilation", implying we should JIT it.
    jit_vmapped_predict = jax.jit(vmapped_predict)

    def run_vmap_benchmark():
        print("Compiling and warming up Vmap...")
        # Warmup on small subset
        warmup_subset_size = 10
        _ = jit_vmapped_predict(mu[:warmup_subset_size], lambda_par[:warmup_subset_size]).block_until_ready()
        
        print(f"Running Vmap on {N_VOXELS} voxels...")
        start_time = time.perf_counter()
        
        # Run on full dataset
        res = jit_vmapped_predict(mu, lambda_par)
        res.block_until_ready()
        
        end_time = time.perf_counter()
        return end_time - start_time

    # 5. Execute Benchmarks
    # Run Serial
    time_serial = run_serial_loop()
    
    # Run Vmap
    time_vmap = run_vmap_benchmark()
    
    # 6. Output Results
    print("\nBenchmark Results:")
    print("=" * 60)
    print(f"{'Method':<25} | {'Time (s)':<15} | {'Voxels/sec':<15}")
    print("-" * 60)
    print(f"{'Serial Loop':<25} | {time_serial:<15.5f} | {N_VOXELS/time_serial:<15.1f}")
    print(f"{'JAX Vmap (Parallel)':<25} | {time_vmap:<15.5f} | {N_VOXELS/time_vmap:<15.1f}")
    print("-" * 60)
    speedup = time_serial / time_vmap
    print(f"Speedup Factor: {speedup:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
