
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from dmipy_jax.inference.flows import FlowNetwork

# 1. Define Model: 1D Bi-Exponential
# S(b) = f * exp(-b * D1) + (1-f) * exp(-b * D2)
# Parameters: f, D1, D2. 
# Constraints: 0 < f < 1, D1 > 0, D2 > 0.
# Degeneracy: If we swap (f, D1) with (1-f, D2), signal is identical.

class BiExpModel(eqx.Module):
    # We parameterize in unconstrained space
    # f_logit -> sigmoid(f)
    # log_D1 -> exp(D1)
    # log_D2 -> exp(D2)
    
    def __call__(self, theta, bvals):
        # theta: [f_logit, log_D1, log_D2]
        f = jax.nn.sigmoid(theta[0])
        D1 = jnp.exp(theta[1])
        D2 = jnp.exp(theta[2])
        
        S = f * jnp.exp(-bvals * D1) + (1 - f) * jnp.exp(-bvals * D2)
        return S

def main():
    print("Running Degeneracy Demo (Bi-Exponential)...")
    
    # 2. Generate Data
    # True Params: f=0.3, D1=1.0, D2=3.0 (arbitrary units)
    # Degenerate Solution: f=0.7, D1=3.0, D2=1.0
    
    true_f = 0.3
    true_D1 = 1.0
    true_D2 = 3.0
    
    # Convert to unconstrained
    true_theta = jnp.array([
        jnp.log(true_f / (1 - true_f)), # logit ~ -0.84
        jnp.log(true_D1),               # log(1) = 0
        jnp.log(true_D2)                # log(3) ~ 1.09
    ])
    
    bvals = jnp.linspace(0, 5, 20)
    model = BiExpModel()
    
    signal_noiseless = model(true_theta, bvals)
    
    # Add noise
    key = jax.random.PRNGKey(42)
    noise_sigma = 0.05
    signal_noisy = signal_noiseless + noise_sigma * jax.random.normal(key, signal_noiseless.shape)
    
    print(f"Data Generated. Noise Sigma: {noise_sigma}")
    
    # 3. Fit Normalizing Flow
    # We want p(theta | y).
    # We treat 'y' (signal) as context.
    # Flow learns to map N(0,I) -> Theta conditioned on y.
    
    # Setup Flow
    key_flow, key_train = jax.random.split(key)
    flow = FlowNetwork(key_flow, n_layers=4, n_dim=3, n_context=len(bvals))
    
    print("Flow structure:")
    print(eqx.tree_pformat(flow))
    
    # Check leaves types
    leaves, _ = jax.tree_util.tree_flatten(flow)
    print(f"Leaves types: {[type(l) for l in leaves]}")
    
    # Partition model into learnable params and static structure
    params, static = eqx.partition(flow, eqx.is_array)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    @eqx.filter_jit
    def loss_fn(params, static, key, context):
        flow_model = eqx.combine(params, static)
        
        # Sample theta from Flow
        z_sample = jax.random.normal(key, (3,))
        theta_sample = flow_model.sample(key, context, n_samples=1) 
        # flow.sample returns (N, D).
        theta_sample = theta_sample[0]
        
        # Prior
        log_prior = jax.scipy.stats.norm.logpdf(theta_sample, scale=3.0).sum()
        
        # Likelihood
        pred = model(theta_sample, bvals) # bvals captured from outer scope? Yes.
        log_like = jax.scipy.stats.norm.logpdf(signal_noisy, loc=pred, scale=noise_sigma).sum()
        
        log_q = flow_model.log_prob(theta_sample, context)
        
        return log_q - log_like - log_prior
        
    @eqx.filter_jit
    def step(params, static, opt_state, k):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, static, k, signal_noisy)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss
        
    print("Training Flow...")
    n_steps = 2000
    for i in range(n_steps):
        key_train, k = jax.random.split(key_train)
        params, opt_state, l = step(params, static, opt_state, k)
        if i % 200 == 0:
            print(f"Step {i}, Loss: {l:.4f}")
    
    # Recombine for sampling
    flow = eqx.combine(params, static)
            
    # 4. Analyze Results
    print("Sampling Posterior...")
    posterior_samples = flow.sample(key, signal_noisy, n_samples=1000)
    print("Samples generated. Shape:", posterior_samples.shape)
    
    # Check for modes
    f_samples = jax.nn.sigmoid(posterior_samples[:, 0])
    
    # Count peaks in f
    f_mean = jnp.mean(f_samples)
    f_std = jnp.std(f_samples)
    print(f"Posterior f: Mean={f_mean:.3f}, Std={f_std:.3f}")
    
    hist, bins = jnp.histogram(f_samples, bins=10, range=(0,1))
    print(f"Structure of f (Histogram): {hist}") 
    # If degenerate, we should see mass around 0.3 AND 0.7.
    
    # Save dummy plot data? Or just print for now.
    
    # Also check if D1/D2 swap
    # Scatter D1 vs D2?
    
if __name__ == "__main__":
    main()
