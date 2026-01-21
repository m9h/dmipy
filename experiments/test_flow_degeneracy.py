
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from dmipy_jax.inference.flows import FlowNetwork

def main():
    print("=== Flow Network Verification ===")
    key = jax.random.PRNGKey(0)
    
    # 1. Synthetic Data: Bimodal distribution
    # Just a simple 2D distribution to test density estimation
    # Mixture of two Gaussians: at (-1, -1) and (1, 1)
    
    n_samples = 1000
    k1, k2, k3 = jax.random.split(key, 3)
    
    s1 = jax.random.normal(k1, (n_samples // 2, 2)) * 0.3 - 1.0
    s2 = jax.random.normal(k2, (n_samples // 2, 2)) * 0.3 + 1.0
    data = jnp.concatenate([s1, s2], axis=0)
    
    # Dummy context (signal) - for this test we ignore context conditioning
    # or just pass zeros.
    context = jnp.zeros((n_samples, 5))
    
    # 2. Init Flow
    # 2D flow, 2 layers
    flow = FlowNetwork(k3, n_layers=2, n_dim=2, n_context=5)
    
    # 3. Train Loop
    optimizer = optax.adam(1e-3)
    
    # Filter params
    params, static = eqx.partition(flow, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    
    @eqx.filter_jit
    def loss_fn(params, static, x, c):
        model = eqx.combine(params, static)
        # Minimize Negative Log Likelihood
        # log_prob returns scalar sum log prob for the batch? 
        # No, my implementation summed over batch? 
        # Let's check: log_prob calls sum() at end.
        
        # We need batch log prob usually.
        # My implementation:
        # log_prob_z = jax.scipy.stats.norm.logpdf(x).sum()
        # This sums over EVERYTHING. Correct for scalar loss.
        
        # Vmap model.log_prob over batch
        lp = jax.vmap(model.log_prob)(x, c)
        return -jnp.mean(lp) # Average NLL
        
    @eqx.filter_jit
    def step(params, static, opt_state, x, c):
        loss, grads = jax.value_and_grad(loss_fn)(params, static, x, c)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    print("Training Flow...")
    for i in range(100):
        # Batch training (full batch here)
        params, opt_state, loss = step(params, static, opt_state, data, context)
        if i % 20 == 0:
            print(f"Epoch {i}: NLL = {loss:.4f}")
            
    print(f"Final NLL: {loss:.4f}")
    
    # Since we can't sample (inverse is identity placeholder), we rely on NLL decrease.
    # Initial NLL for N(0,I) on this data (approx):
    # Data is at +/- 1. Normal is at 0. Variance 1.
    # Should be reasonably high.
    # If flow works, NLL should decrease as it adapts to the two modes.
    
    if loss < 2.5: # Heuristic threshold
        print("SUCCESS: Flow is learning (NLL decreased).")
    else:
        print("WARNING: Flow might not be learning well.")

if __name__ == "__main__":
    main()
