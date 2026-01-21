
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import pytest
from dmipy_jax.inference.flows import RationalQuadraticSpline, FlowNetwork

def test_rational_quadratic_spline_invertibility():
    """Verify that RQS is bijective (Inverse(Forward(x)) approx x)."""
    key = jax.random.PRNGKey(42)
    D = 1
    K = 4
    params = jax.random.normal(key, (D, 3*K + 1))
    
    # Random input in range [-2, 2]
    x = jnp.array([1.5]) 
    
    # Instantiate
    rqs = RationalQuadraticSpline(num_bins=K)
    
    # Forward
    z, log_det = rqs(x, params, inverse=False)
    
    # Inverse
    x_rec, inv_log_det = rqs(z, params, inverse=True)
    
    # Check reconstruction
    assert jnp.allclose(x, x_rec, atol=1e-4), f"Reconstruction failed: {x} -> {z} -> {x_rec}"
    
    # Check log det consistency (forward + inverse should sum to ~0)
    # log|dz/dx| + log|dx/dz| = 0
    assert jnp.allclose(log_det + inv_log_det, 0.0, atol=1e-4)

def test_flow_network_shapes():
    """Verify FlowNetwork output shapes."""
    key = jax.random.PRNGKey(0)
    flow = FlowNetwork(key, n_layers=2, n_dim=2, n_context=5)
    
    theta = jnp.ones((2,))
    context = jnp.ones((5,))
    
    # Log Prob
    lp = flow.log_prob(theta, context)
    assert lp.shape == ()
    
    # Sample
    samples = flow.sample(key, context, n_samples=10)
    assert samples.shape == (10, 2)

def test_flow_training_toy_problem():
    """
    Train a Flow to learn a conditional distribution: p(x | c) = N(x; c, 0.1).
    The flow should learn to shift the base distribution N(0,1) by 'c'.
    """
    key = jax.random.PRNGKey(101)
    
    # Setup Flow
    # 1D flow
    flow = FlowNetwork(key, n_layers=3, n_dim=2, n_context=1)
    # We use 2D flow because coupling layers usually split dimensions. 
    # Let's map p(y | c) ~ N( [c, c], 0.1 I )
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(flow, opt_state, batch_x, batch_c):
        def loss_fn(model):
            # Maximize log prob -> Minimize -log prob
            lp = jax.vmap(model.log_prob)(batch_x, batch_c)
            return -jnp.mean(lp)
            
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        updates, opt_state = optimizer.update(grads, opt_state, flow)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    # Train Loop
    n_steps = 500
    batch_size = 128
    
    loss_history = []
    
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        # Generate Data
        # Context c ~ U[-2, 2]
        c = jax.random.uniform(subkey, (batch_size, 1), minval=-2, maxval=2)
        
        # x ~ N(c, 0.1)
        noise = jax.random.normal(subkey, (batch_size, 2)) * 0.1
        x = c + noise # Broadcasts c to (B, 1) -> (B, 2)? No, c is (B,1), x is (B,2).
        
        flow, opt_state, loss = make_step(flow, opt_state, x, c)
        loss_history.append(loss)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            
    # Verification
    # Loss should decrease significantly (negative log likelihood)
    # With sigma=0.1, entropy is low. 
    # log_prob for true model: -0.5*(x-mu)^2/sigma^2 - log(sigma*sqrt(2pi))
    # average expected log_prob ~ -log(0.1) - 0.5 ~ 2.3 - 0.5 = 1.8. 
    # Loss should be around -1.8 * 2 (dims) = -3.6 approx.
    
    print(f"Final Loss: {loss_history[-1]}")
    assert loss_history[-1] < 0.0, "Loss didn't converge to negative values (log prob maximization)"
    
    # Check conditional sampling
    test_c = jnp.array([1.0])
    samples = flow.sample(key, test_c, n_samples=100)
    
    # Mean of samples should be close to c=1.0
    sample_mean = jnp.mean(samples, axis=0)
    print(f"Sample Mean (Target 1.0): {sample_mean}")
    
    assert jnp.allclose(sample_mean, 1.0, atol=0.2)

if __name__ == "__main__":
    test_rational_quadratic_spline_invertibility()
    test_flow_network_shapes()
    test_flow_training_toy_problem()
    print("All Flow Tests Passed!")
