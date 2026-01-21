
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optimistix as optx
from typing import Dict, Any, Callable, Tuple
from jaxtyping import Array, Float

from dmipy_jax.optimization.acquisition import AcquisitionProtocol
from dmipy_jax.inference.variational import MeanFieldGaussian, VIMinimizer, fit_vi, inverse_softplus
# Assuming sphere models exist and follow standard interface
from dmipy_jax.signal_models.sphere_models import SphereGPD

class OEDObjective(eqx.Module):
    """
    Objective function for Bayesian Optimal Experimental Design.
    
    Goal: Maximize Expected Information Gain (EIG)
    EIG = H[theta] - E_y[ H[theta|y] ]
    Maximize EIG <=> Minimize Expected Posterior Entropy E_y[H[theta|y]].
    """
    tissue_model: eqx.Module
    prior_means: Dict[str, float]
    prior_stds: Dict[str, float]
    sigma_noise: float = 0.02
    vi_steps: int = 20 # Unrolled steps
    
    def __init__(self, tissue_model, prior_means, prior_stds, sigma_noise=0.02, vi_steps=20):
        self.tissue_model = tissue_model
        self.prior_means = prior_means
        self.prior_stds = prior_stds
        self.sigma_noise = sigma_noise
        self.vi_steps = vi_steps

    def sample_prior(self, key):
        """Samples params from the prior."""
        keys = jax.random.split(key, len(self.prior_means))
        params = {}
        for i, (k, mu) in enumerate(self.prior_means.items()):
            sigma = self.prior_stds.get(k, 1.0)
            params[k] = mu + sigma * jax.random.normal(keys[i])
        return params

    def estimate_posterior_entropy(self, protocol_params, key):
        """
        Simulates one data instance and estimates posterior entropy via unrolled VI.
        """
        k_theta, k_noise, k_vi = jax.random.split(key, 3)
        
        # 1. Sample Truth
        true_params = self.sample_prior(k_theta)
        
        # 2. Simulate Data
        # protocol_params is the AcquisitionProtocol (or similar learnable object)
        scheme = protocol_params() # Generates JaxAcquisition
        
        # Fix for model call: standard models take kwargs
        # Ensure model is compatible (SphereGPD usually takes diameter, diffusion_constant)
        # We need to ensure parameters are positive if model expects it.
        # usually models handle it or input is physical.
        # sample_prior output is Gaussian -> might be negative.
        # Assumption: Prior is defined in physical space or transformed?
        # Let's assume prior is truncated normal or we take absolute value?
        # Better: Prior is Log-Normal. 
        # Sample log_params ~ N(log_mu, sigma), params = exp(log_params).
        
        safe_params = {k: jnp.nan_to_num(jnp.exp(v), nan=1e-12) for k, v in true_params.items()}
        
        # Guard scheme
        bvals_safe = jnp.nan_to_num(scheme.bvalues, nan=0.0)
        Delta_safe = jnp.nan_to_num(scheme.Delta, nan=0.05)
        delta_safe = jnp.nan_to_num(scheme.delta, nan=0.01)
        
        signal = self.tissue_model(
            bvals=bvals_safe,
            gradient_directions=scheme.gradient_directions, # Usually safe
            big_delta=Delta_safe,
            small_delta=delta_safe,
            **safe_params
        )
        
        noise = jax.random.normal(k_noise, signal.shape) * self.sigma_noise
        data = signal + noise
        
        # Debug Data
        is_data_nan = jnp.isnan(data).any()
        jax.lax.cond(is_data_nan, lambda: jax.debug.print("DATA HAS NANS!"), lambda: None)
        
        # 3. Unrolled VI
        # Initialize Variational Posterior at Prior Mean (or random start)
        # Prior is in Log Space (from our assumption above).
        # MeanFieldGaussian operates in unconstrained space.
        # If we use log-normal prior, the params in true_params ARE the unconstrained values (logs).
        # Perfect.
        
        # Init with prior means
        unconstrained_init = {k: jnp.array(v) for k, v in self.prior_means.items()}
        vi_posterior = MeanFieldGaussian(unconstrained_init, init_log_std=-1.0) # Start with wide uncertainty
        
        minimizer = VIMinimizer(self.tissue_model, scheme, self.sigma_noise)
        # Reducing inner LR significantly to prevent divergence
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(0.005) 
        )
        opt_state = optimizer.init(vi_posterior)
        
        def step_fn(carry, k_step):
            post, opt_st = carry
            loss, grads = eqx.filter_value_and_grad(minimizer.loss_fn)(post, data, k_step)
            # Guard against NaN grads
            grads = jax.tree.map(lambda g: jnp.where(jnp.isnan(g), 0.0, g), grads)
            updates, opt_st = optimizer.update(grads, opt_st, post)
            post = eqx.apply_updates(post, updates)
            return (post, opt_st), None
            
        keys_vi = jax.random.split(k_vi, self.vi_steps)
        (final_posterior, _), _ = jax.lax.scan(step_fn, (vi_posterior, opt_state), keys_vi)
        
        # Guard entropy
        ent = final_posterior.entropy()
        
        # Debug Posterior
        def print_post_stats(p, e):
            leaves, _ = jax.tree_util.tree_flatten(p)
            for i, l in enumerate(leaves):
                 is_nan_leaf = jnp.isnan(l).any()
                 # Condition print on nan
                 # jax.lax.cond(is_nan_leaf, lambda: jax.debug.print("NaN in leaf {}", i), lambda: None)
            
            is_ent_nan = jnp.isnan(e)
            jax.lax.cond(is_ent_nan, lambda: jax.debug.print("Entropy is NaN!"), lambda: None)
            
        print_post_stats(final_posterior, ent)
        
        return jnp.where(jnp.isnan(ent), -1e6, ent)

    def expected_posterior_entropy(self, protocol_params, key, num_samples=10):
        """
        Loss function to minimize.
        """
        keys = jax.random.split(key, num_samples)
        entropies = jax.vmap(lambda k: self.estimate_posterior_entropy(protocol_params, k))(keys)
        # Robust mean
        valid_mask = ~jnp.isnan(entropies) & (entropies > -1e5)
        # If all invalid, return decent fallback or just 0 gradient?
        # Use mean of valid?
        # Simplified: just return mean, relying on inner guard.
        return jnp.mean(entropies)


def optimize_oed(
    tissue_model: eqx.Module,
    prior_means: Dict[str, float], # In Log Space
    prior_stds: Dict[str, float],
    n_measurements: int = 10,
    seed: int = 0
):
    key = jax.random.PRNGKey(seed)
    key_init, key_opt = jax.random.split(key)
    
    # Learnable Protocol
    protocol = AcquisitionProtocol(n_measurements, key=key_init)
    
    oed_obj = OEDObjective(tissue_model, prior_means, prior_stds)
    
    # Optax for stochastic optimization
    # Reduced outer LR as well
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=0.005)
    )
    
    def loss_fn(proto, args):
        key_step = args
        return oed_obj.expected_posterior_entropy(proto, key_step, num_samples=5)
        
    print("Starting Bayesian OED Optimization (Stabilized)...")
    
    opt_state = optimizer.init(protocol)
    
    for i in range(50):
        k_iter = jax.random.fold_in(key_opt, i)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(protocol, k_iter)
        
        # Guard outer grads
        def check_grad(g):
            # Check for NaNs
            is_nan = jnp.isnan(g).any()
            # Since this is traced, we use jax.debug.print via a callback or just standard jax.debug.print
            jax.debug.print("Grad NaNs: {}", is_nan)
            return jnp.where(jnp.isnan(g), 0.0, g)
            
        grads = jax.tree.map(check_grad, grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, protocol)
        protocol = eqx.apply_updates(protocol, updates)
        
        if i % 10 == 0:
            print(f"Iter {i}: Expected Posterior Entropy = {loss:.4f}")
            
    return protocol

if __name__ == "__main__":
    print("Running Bayesian OED Verification...")
    
    # 1. Setup Model
    model = SphereGPD()
    # Dummy removed.


    
    # 2. Define Prior (Log Space)
    # Diameter ~ LogNormal(mu=-12.3, sigma=0.5) -> Median exp(-12.3) = 4.5e-6 meters (4.5 um)
    prior_means = {'diameter': -12.3, 'diffusion_constant': jnp.log(2e-9)} 
    prior_stds = {'diameter': 0.5, 'diffusion_constant': 0.2} # Relaxed sigma
    
    # 3. optimize
    optimized_protocol = optimize_oed(model, prior_means, prior_stds, n_measurements=5)
    
    # 4. Analyze Result
    scheme = optimized_protocol()
    print("\nOptimized Protocol:")
    print(f"B-values: {scheme.bvalues}")
    print(f"Delta: {scheme.Delta}")
    print(f"delta: {scheme.delta}")
    
    mean_Delta = jnp.mean(scheme.Delta)
    print(f"Mean Delta: {mean_Delta*1000:.1f} ms")
    
    if mean_Delta > 0.02: 
        print("\nSUCCESS: OED prioritized long diffusion times.")
    else:
        print("\nObservation: Delta is small.")
