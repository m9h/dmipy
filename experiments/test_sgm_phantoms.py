
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax
import matplotlib.pyplot as plt
from dmipy_jax.simulation.phantoms import SGM

def main():
    print("=== SGM Verification Experiment ===")
    key = jax.random.PRNGKey(42)
    
    # 1. Synthetic Data: "Banana" or mixture
    # Mixture of 3 Gaussians in 2D
    n_samples = 2000
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    s1 = jax.random.normal(k1, (n_samples//3, 2)) * 0.3 + jnp.array([1.0, 1.0])
    s2 = jax.random.normal(k2, (n_samples//3, 2)) * 0.3 + jnp.array([-1.0, -1.0])
    s3 = jax.random.normal(k3, (n_samples//3, 2)) * 0.3 + jnp.array([1.0, -1.0])
    
    train_data = jnp.concatenate([s1, s2, s3], axis=0)
    
    # 2. Initialize SGM
    data_dim = 2
    model = SGM(k4, data_dim)
    
    model = SGM(k4, data_dim)
    
    optimizer = optax.adam(1e-3)
    
    # Filter trainable params
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    
    # 3. Training Loop
    @eqx.filter_jit
    def step(params, static, opt_state, batch, key):
        model = eqx.combine(params, static)
        
        # Sample random times t ~ U[0, 1] (avoiding 0 strictly for stability)
        # Typically U[epsilon, 1]
        epsilon = 1e-5
        t = jax.random.uniform(key, (batch.shape[0],), minval=epsilon, maxval=1.0)
        
        # Loss: vmapped over batch
        k_batch = jax.random.split(key, batch.shape[0])
        
        # loss_fn(model, x0, t, key)
        # vmap over x0, t, key
        # Note: model contains BOTH params and static now, combine happened above.
        
        # We need gradients wrt params
        def compute_loss(p):
            m = eqx.combine(p, static)
            losses = jax.vmap(m.loss_fn, in_axes=(None, 0, 0, 0))(m, batch, t, k_batch)
            return jnp.mean(losses)
            
        loss, grads = eqx.filter_value_and_grad(compute_loss)(params)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    print("Training SGM...")
    n_epochs = 2000
    batch_size = 256
    
    # Simple full batch or random batch?
    # Random batch
    for i in range(n_epochs):
        k_step = jax.random.fold_in(key, i)
        idx = jax.random.randint(k_step, (batch_size,), 0, train_data.shape[0])
        batch = train_data[idx]
        
        # Use subkey for noise
        k_noise = jax.random.fold_in(k_step, 999)
        params, opt_state, loss = step(params, static, opt_state, batch, k_noise)
        
        if i % 200 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
            
    # 4. Sampling
    print("Sampling...")
    k_sample = jax.random.fold_in(key, 9999)
    # Reconstruct full model
    model = eqx.combine(params, static)
    samples = model.sample(k_sample, n_samples=500)
    
    # 5. Visualization (if possible)
    # Check bounds
    print("Generated Samples Mean:", jnp.mean(samples, axis=0))
    print("Train Data Mean:", jnp.mean(train_data, axis=0))
    
    # MMD or simple check: samples should be clusterd around means
    # Clusters at (1,1), (-1,-1), (1,-1)
    
    # Count samples near modes
    def count_near(center):
        return jnp.sum(jnp.linalg.norm(samples - center, axis=1) < 0.8)
        
    c1 = count_near(jnp.array([1.0, 1.0]))
    c2 = count_near(jnp.array([-1.0, -1.0]))
    c3 = count_near(jnp.array([1.0, -1.0]))
    
    print(f"Counts near modes: {c1}, {c2}, {c3}")
    
    if c1 > 20 and c2 > 20 and c3 > 20:
        print("SUCCESS: SGM generated samples covering all 3 modes.")
        
        plt.scatter(train_data[:,0], train_data[:,1], alpha=0.1, label='Train')
        plt.scatter(samples[:,0], samples[:,1], alpha=0.5, label='Generated')
        plt.legend()
        plt.savefig("experiments/sgm_result.png")
        print("Plot saved.")
    else:
        print("WARNING: SGM failed to cover modes.")

if __name__ == "__main__":
    main()
