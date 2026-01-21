
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
from dmipy_jax.simulation.phantoms import SGM

def main():
    print("=== SGM 'Killer Example': Manifold Learning for Virtual Populations ===")
    print("Concept: Learn strict biological constraints (e.g., diameter vs density) from sparse data.")
    
    key = jax.random.PRNGKey(0)
    
    # 1. Define "Biological Manifold"
    # Example: Axon Diameter (d) vs Intracellular Volume Fraction (f)
    # Biophysical constraint: Packing geometry limits f based on d.
    # Let's model a correlation: f ~ 0.7 * (1 - exp(-d))
    # This represents that larger axons pack differently or just a correlation.
    # Or simple non-linear correlation: y = x^2 (Parabola)
    
    print("Generating sparse 'Patient Data' from ground truth manifold...")
    n_data = 1000 # More data for stability
    k_data, k_noise = jax.random.split(key)
    
    # x: 'Parameter 1' (e.g., Diameter index -2 to 2)
    x = jax.random.uniform(k_data, (n_data, 1), minval=-2.0, maxval=2.0)
    
    # y: 'Parameter 2' (correlated) = x^2 - 1.0 + noise
    noise = jax.random.normal(k_noise, (n_data, 1)) * 0.1
    y = x**2 - 1.0 + noise
    
    train_data = jnp.concatenate([x, y], axis=1) # (N, 2)
    
    # 2. Train SGM to learn this manifold
    print("Training Score-Based Model to learn the manifold...")
    model = SGM(jax.random.PRNGKey(42), data_dim=2)
    
    optimizer = optax.adam(3e-4) # Lower LR
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    
    @eqx.filter_jit
    def step(params, static, opt_state, batch, k):
        model = eqx.combine(params, static)
        # Loss: vmap over batch
        k_batch = jax.random.split(k, batch.shape[0])
        # Random time t per sample
        t = jax.random.uniform(k, (batch.shape[0],), minval=1e-5, maxval=1.0)
        
        # We need gradients wrt params
        def compute_loss(p):
            m = eqx.combine(p, static)
            losses = jax.vmap(m.loss_fn, in_axes=(None, 0, 0, 0))(m, batch, t, k_batch)
            return jnp.mean(losses)
            
        loss, grads = eqx.filter_value_and_grad(compute_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    n_steps = 4000
    batch_size = 256
    k_train = jax.random.PRNGKey(1)
    
    for i in range(n_steps):
        k_step, k_train = jax.random.split(k_train)
        idx = jax.random.randint(k_step, (batch_size,), 0, n_data)
        batch = train_data[idx]
        
        params, opt_state, loss = step(params, static, opt_state, batch, k_step)
        
        if i % 500 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            
    # 3. Generate "Infinite" Virtual Population
    print("Generating 'Virtual Cohort' (1000 samples)...")
    model = eqx.combine(params, static)
    k_gen = jax.random.PRNGKey(999)
    # Generate 1000 samples
    samples = model.sample(k_gen, n_samples=1000)
    
    # 4. Visualization/Analysis
    # Check if samples lie on the parabola y = x^2 - 1
    # MSE of samples against manifold
    x_gen = samples[:, 0]
    y_gen = samples[:, 1]
    y_pred = x_gen**2 - 1.0
    mse = jnp.mean((y_gen - y_pred)**2)
    print(f"Manifold Error (MSE of generated samples vs Truth): {mse:.4f}")
    
    if mse < 0.2: # Allow variance
        print("SUCCESS: SGM successfully learned the biological constraint!")
        try:
             # ASCII Plot
            print("\nASCII Scatter Plot (Approx):")
            # Simple grid
            grid = [[' ' for _ in range(40)] for _ in range(20)]
            
            # Map -2.5 to 2.5 on x (40 bins), -1.5 to 3.5 on y (20 bins)
            def plot_pt(x, y, char):
                bx = int((x + 2.5) / 5.0 * 40)
                by = int((y + 1.5) / 5.0 * 20)
                if 0 <= bx < 40 and 0 <= by < 20:
                    grid[19-by][bx] = char
            
            # Plot train
            for i in range(0, n_data, 5): # subset
                plot_pt(train_data[i,0], train_data[i,1], '.')
                
            # Plot gen
            for i in range(0, 1000, 20): # subset
                plot_pt(samples[i,0], samples[i,1], 'O')
                
            print("Legend: (.)=Real Data, (O)=Generated Virtual Patient")
            print("-" * 42)
            for row in grid:
                print("|" + "".join(row) + "|")
            print("-" * 42)
            
            plt.figure(figsize=(8,6))
            plt.scatter(train_data[:,0], train_data[:,1], alpha=0.3, label='Real Patient Data', color='blue')
            plt.scatter(samples[:,0], samples[:,1], alpha=0.3, label='Generated Virtual Cohort', color='red')
            x_range = np.linspace(-2.5, 2.5, 100)
            plt.plot(x_range, x_range**2 - 1.0, 'k--', label='True Biological Manifold')
            plt.legend()
            plt.title("SGM Generative Phantom: Learning Biological Constraints")
            plt.savefig("experiments/sgm_killer_example.png")
            print("Plot saved to experiments/sgm_killer_example.png")
        except Exception as e:
            print(f"Plotting failed: {e}")
    else:
        print("WARNING: SGM failed to learn manifold well.")

if __name__ == "__main__":
    main()
