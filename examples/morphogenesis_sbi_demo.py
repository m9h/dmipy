"""
Morphogenesis SBI Inference Demo.
Trains a normalizing flow to infer growth parameters.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
try:
    import flowjax.distributions as distributions
    import flowjax.flows as flows
    from flowjax.train import train_flow
    FLOWJAX_AVAILABLE = True
except ImportError:
    FLOWJAX_AVAILABLE = False

from dmipy_jax.morphogenesis.inference import MorphoSimulator, generate_training_data

def run_sbi_demo():
    if not FLOWJAX_AVAILABLE:
        print("FlowJAX not available. Skipping training demo.")
        return

    key = jr.PRNGKey(42)
    sim_key, train_key, flow_key = jr.split(key, 3)
    
    # 1. Setup Simulator (Small grid for demo speed)
    simulator = MorphoSimulator(grid_shape=(8, 8, 8), dx=1.0, n_iters=20)
    
    # 2. Generate Training Data
    print("Generating training data (50 simulations)...")
    # Using small N for speed, real runs would need 1000+
    thetas, xs = generate_training_data(simulator, n_samples=50, key=sim_key)
    
    # 3. Setup Normalizing Flow (NPE)
    # theta dim = 3, x dim = 5
    print("Initializing FlowJAX Neural Spline Flow...")
    flow = flows.masked_autoregressive_flow(
        flow_key, 
        base_dist=distributions.Normal(jnp.zeros(3)),
        transformer=flows.RationalQuadraticSpline(knots=8, interval=3.0),
        cond_dim=5
    )
    
    # 4. Train
    print("Training flow...")
    flow, losses = train_flow(
        flow, thetas, xs, 
        learning_rate=1e-3, 
        num_epochs=10, 
        batch_size=10,
        key=train_key
    )
    
    # 5. Perform Inference on "Target" observation
    # Define a ground truth target
    theta_target = jnp.array([1.2, 0.5, -0.5])
    print(f"\nGround Truth Target: {theta_target}")
    
    x_obs = simulator(theta_target, jr.PRNGKey(0))
    print(f"Target Summary Stats: {x_obs}")
    
    # Sample from posterior
    samples = flow.sample(jr.PRNGKey(1), (1000,), condition=x_obs)
    posterior_mean = jnp.mean(samples, axis=0)
    posterior_std = jnp.std(samples, axis=0)
    
    print("\n--- Inferred Parameters ---")
    print(f"Posterior Mean: {posterior_mean}")
    print(f"Posterior Std:  {posterior_std}")
    
    diff = jnp.abs(posterior_mean - theta_target)
    print(f"\nError: {diff}")
    print("\nSBI Pipeline Complete!")

if __name__ == "__main__":
    run_sbi_demo()
