
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from dmipy_jax.core.surrogate import PolynomialChaosExpansion
from dmipy_jax.signal_models.stick import Stick

def stick_model_jax(d_par, bvals, gradient_directions):
    """
    JAX implementation of the Stick model wrapper for comparison.
    d_par: (N, 1) or (N,)
    """
    # Stick expects: bvals, gradient_directions, mu, lambda_par
    # We fix mu to [0, 0, 1] as in the pygpc script
    
    stick = Stick(mu=jnp.array([0.0, 0.0, 1.0]), lambda_par=d_par)
    
    # Stick call signature: (bvals, gradient_directions, **kwargs)
    # But Stick.__call__ is vmapped over parameters efficiently? 
    # Actually, the signal models usually expect scalar parameters or batched parameters if vmapped.
    # The Stick class provided in the view_file accepts `mu` and `lambda_par`.
    
    # We need to broadcast bvals/gradients.
    # But usually we vmap over parameters.
    
    def single_eval(d):
        return stick(bvals, gradient_directions, lambda_par=d)
    
    # Vmap over the batch of d_par
    return jax.vmap(single_eval)(d_par)

def main():
    print("Starting Comparison Benchmark...")
    
    # 1. Load PyGPC Results
    try:
        with open("pygpc_results.json", "r") as f:
            pygpc_results = json.load(f)
    except FileNotFoundError:
        print("Error: pygpc_results.json not found. Run benchmark_pygpc_script.py first.")
        return

    print("Loaded PyGPC results.")
    
    # 2. Setup Dmipy-JAX Problem
    # Replicate the setup
    d_par_min = pygpc_results["parameters"]["d_par_min"]
    d_par_max = pygpc_results["parameters"]["d_par_max"]
    bvals = jnp.array(pygpc_results["bvals"])
    gradient_directions = jnp.array(pygpc_results["gradient_directions"])
    
    # Generate Integration Points (Training Data)
    # PyGPC used a grid. We need to define how we fit.
    # For fair comparison, we can use a similar number of points or just a standard grid.
    # Let's use a standard grid appropriate for Order 4.
    # For 1D, order 4 needs 5 points for exact integration of degree 9 (2n+1).
    # But to fit a degree 4 polynomial, we need at least 5 points.
    # Let's use 10 points for safety and robust float fitting.
    
    key = jax.random.PRNGKey(0)
    n_samples = 50 # Generous number of samples
    # We fit on Uniform [0.1e-9, 3.0e-9]
    # But dmipy-jax surrogate usually assumes Uniform [-1, 1].
    # We need to map [0.1e-9, 3.0e-9] -> [-1, 1].
    
    # Actual phyiscal values
    d_par_samples_phys = jax.random.uniform(key, (n_samples, 1), minval=d_par_min, maxval=d_par_max)
    
    # Standardized values (for input to surrogate)
    # x = 2 * (d - min) / (max - min) - 1
    d_par_samples_std = 2 * (d_par_samples_phys - d_par_min) / (d_par_max - d_par_min) - 1.0
    
    # Calculate True Values
    # Note: Stick model expects physical values
    # Reshape d_par_samples_phys for vmap: (N, 1) -> (N,) if needed, or stick handles it.
    # Our wrapper expects (N,) or (N, 1). 
    # stick_model_jax vmaps over d_par.
    
    values = stick_model_jax(d_par_samples_phys[:, 0], bvals, gradient_directions)
    
    # 3. Fit Fit Dmipy-JAX Surrogate
    start_time = time.time()
    
    surrogate = PolynomialChaosExpansion.fit(
        parameters=d_par_samples_std,
        values=values,
        distributions=['Uniform'],
        total_order=4
    )
    
    end_time = time.time()
    jax_time = end_time - start_time
    
    # 4. Compare Results
    # PyGPC coefficients are likely for Legendre polynomials on [-1, 1] as well 
    # (if it maps domain correctly, which it usually does in gPC).
    
    pygpc_coeffs = np.array(pygpc_results["coefficients"]) # Shape (N_basis, N_outputs)
    jax_coeffs = surrogate.coefficients # Shape (N_basis, N_outputs)
    
    # Align Coefficients
    # Need to match basis indices.
    pygpc_indices = np.array(pygpc_results["basis_indices"])
    jax_indices = surrogate.basis_indices
    
    # Create mapping
    # Since it's 1D, should be simple sort
    # Both are likely sorted by degree, but let's be safe.
    
    # We assume 1D for now
    pygpc_order = pygpc_indices[:, 0]
    jax_order = jax_indices[:, 0]
    
    print("\nComparison Results:")
    print(f"{'Order':<6} | {'PyGPC Coeff (Mean)':<20} | {'JAX Coeff (Mean)':<20} | {'Diff':<20}")
    print("-" * 75)
    
    max_diff = 0.0
    
    for i in range(5): # Order 0 to 4
        # Find index in arrays
        idx_pygpc = np.where(pygpc_order == i)[0]
        idx_jax = np.where(jax_order == i)[0]
        
        if len(idx_pygpc) > 0 and len(idx_jax) > 0:
            c_pygpc = pygpc_coeffs[idx_pygpc[0]]
            c_jax = jax_coeffs[idx_jax[0]]
            
            # Compare mean of coefficients across b-values/directions
            # (Coefficients are vectors of size N_measurements)
            
            # Let's compare the norm of difference
            diff = np.mean(np.abs(c_pygpc - c_jax))
            max_diff = max(max_diff, diff)
            
            print(f"{i:<6} | {np.mean(c_pygpc):<20.6e} | {np.mean(c_jax):<20.6e} | {diff:<20.6e}")
            
    print("-" * 75)
    print(f"Max Coefficient Difference: {max_diff:.6e}")
    
    # 5. Performance Comparison
    pygpc_time = pygpc_results["time_elapsed"]
    print(f"\nPerformance:")
    print(f"PyGPC Time: {pygpc_time:.4f} s")
    print(f"JAX Time:   {jax_time:.4f} s")
    print(f"Speedup:    {pygpc_time / jax_time:.2f}x")
    
    # 6. Prediction Check
    # Verify JAX surrogate prediction matches JAX model
    validation_params = jnp.array([[0.0]]) # Center of domain (std)
    validation_phys = (validation_params + 1.0) / 2.0 * (d_par_max - d_par_min) + d_par_min
    
    true_val = stick_model_jax(validation_phys[:, 0], bvals, gradient_directions)
    pred_val = surrogate(validation_params)
    
    mse = jnp.mean((true_val - pred_val)**2)
    print(f"\nValidation MSE (Surrogate vs True): {mse:.6e}")
    
if __name__ == "__main__":
    main()
