
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.gaussian import G2Zeppelin, G1Ball
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def main():
    print("Tutorial 5: Spherical Mean Technique (SMT)")
    
    # SMT fits the "Powder Average" signal to estimate microscopic features 
    # independent of orientation dispersion.
    
    # 1. Simulate anisotropic signal (Ground Truth)
    # We simulate a crossing fiber or dispersed signal
    # But for simplicity, let's just simulate a single stick with some orientation.
    # When we take the spherical mean, the orientation dependence should vanish.
    
    # Acquisition: High Angular Resolution to get good mean
    bvals_shell = jnp.array([1000.0, 2000.0, 3000.0]) * 1e6
    n_dirs = 64
    
    # Use helper for directions if available, otherwise random
    key = jax.random.PRNGKey(42)
    vecs = jax.random.normal(key, (n_dirs, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=1, keepdims=True)
    
    bvals = jnp.kron(bvals_shell, jnp.ones(n_dirs))
    bvecs = jnp.kron(jnp.ones(3), vecs).reshape(-1, 3) # Wait, shape mismatch logic
    # Just repeat vector set 3 times
    bvecs = jnp.tile(vecs, (3, 1))
    
    acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)
    
    # Model: Stick
    stick = C1Stick()
    lambda_par = 2.0e-9
    mu = jnp.array([1.0, 0.0, 0.0]) # X-axis
    
    # Simulate
    S_aniso = stick(acq.bvalues, acq.gradient_directions, mu=mu, lambda_par=lambda_par)
    
    # 2. Compute Spherical Mean (Powder Average)
    # Group by shell
    unique_b = jnp.unique(bvals_shell)
    S_mean = []
    
    # Naive averaging (assuming uniform sampling)
    for b in unique_b:
        mask = jnp.isclose(acq.bvalues, b)
        S_shell = S_aniso[mask]
        S_mean.append(jnp.mean(S_shell))
        
    S_mean = jnp.array(S_mean)
    
    print("Spherical Means:", S_mean)
    
    # 3. Fit SMT Model (Analytic Powder Average of Stick)
    # Powder(Stick) = sqrt(pi) / (2 * sqrt(b * lambda_par)) * erf(sqrt(b * lambda_par))
    # Approximation for high b: decay ~ 1/sqrt(b)
    
    from jax.scipy.special import erf
    
    def powder_stick(bval, diff):
        # Limit diff -> 0?
        arg = bval * diff
        # Avoid div by zero
        factor = jnp.sqrt(jnp.pi) / (2 * jnp.sqrt(arg + 1e-12))
        res = factor * erf(jnp.sqrt(arg))
        return jnp.where(bval < 1e3, 1.0, res) # Handle b=0 safely
        
    # Let's check validity of this formula vs numerical mean
    # Or optimize lambda_par to match S_mean
    
    def loss_fn(d_est):
        pred = powder_stick(unique_b, d_est)
        return jnp.mean((pred - S_mean)**2)
        
    # Optimization (Scalar)
    # Grid search for D
    test_ds = jnp.linspace(0.1e-9, 3.0e-9, 100)
    losses = jax.vmap(loss_fn)(test_ds)
    best_d = test_ds[jnp.argmin(losses)]
    
    print(f"Ground Truth Lambda_Par: {lambda_par}")
    print(f"Estimated from SMT: {best_d}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(unique_b, S_mean, 'ko', label='Spherical Mean Data')
    plt.plot(unique_b, powder_stick(unique_b, lambda_par), 'b-', label='Ideal Powder Stick (GT)')
    plt.plot(unique_b, powder_stick(unique_b, best_d), 'r--', label='Fitted SMT')
    plt.xlabel('b-value (s/m^2)')
    plt.ylabel('Signal')
    plt.title('Tutorial 5: SMT (Powder Average Fitting)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tutorial_5_output.png')
    print("Saved SMT plot.")

if __name__ == "__main__":
    main()
