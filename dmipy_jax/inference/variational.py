import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, Float, PyTree
from typing import Callable, Any, Dict, Optional, Tuple

class MeanFieldGaussian(eqx.Module):
    """
    Mean-Field Gaussian Variational Posterior.
    
    Approximates p(theta | y) as a product of independent Normals:
    q(theta) = Product_i N(theta_i | mu_i, sigma_i)
    
    Parameters are parameterized in unconstrained space.
    """
    # Learnable parameters (unconstrained)
    means: Dict[str, Float[Array, "1"]]
    log_stds: Dict[str, Float[Array, "1"]]
    
    def __init__(self, init_params: Dict[str, float], init_log_std: float = -2.0):
        """
        Args:
            init_params: Dictionary of initial mean values for parameters.
            init_log_std: Initial log standard deviation (default: -2.0 -> sigma ~ 0.135).
        """
        self.means = {k: jnp.array(v) for k, v in init_params.items()}
        self.log_stds = {k: jnp.full_like(jnp.array(v), init_log_std) for k, v in init_params.items()}

    def sample(self, key: jax.Array) -> Tuple[Dict[str, float], Float[Array, "num_params"]]:
        """
        Sample from the posterior using the reparameterization trick.
        
        Returns:
            samples_dict: Dictionary of sampled parameters (for the model).
            log_q: Log probability density of the sample log q(theta).
        """
        keys = jax.random.split(key, len(self.means))
        
        samples_dict = {}
        log_prob_sum = 0.0
        
        for i, (k, mean) in enumerate(self.means.items()):
            log_std = jnp.clip(self.log_stds[k], -10.0, 2.0) # Clamp sigma between ~4.5e-5 and 7.4
            std = jnp.exp(log_std)
            eps = jax.random.normal(keys[i], mean.shape)
            
            val = mean + std * eps
            samples_dict[k] = val
            
            # Log probability of a gaussian:
            # -0.5 * log(2pi) - log(std) - 0.5 * ((x - mu)/std)^2
            # Here ((x - mu)/std)^2 is just eps^2
            log_p = -0.5 * jnp.log(2 * jnp.pi) - log_std - 0.5 * (eps ** 2)
            log_prob_sum += jnp.sum(log_p)
            
        return samples_dict, log_prob_sum

    def entropy(self) -> Float[Array, ""]:
        """
        Compute the entropy of the approximation: H[q] = -E[log q].
        For a Gaussian: H = 0.5 * log(2pi * e * sigma^2)
        """
        entropy_sum = 0.0
        for k in self.log_stds:
            log_std = jnp.clip(self.log_stds[k], -10.0, 2.0)
            # H = 0.5 (1 + log(2pi) + 2*log_std)
            h = 0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std)
            entropy_sum += jnp.sum(h)
        return entropy_sum

class VIMinimizer(eqx.Module):
    """
    Minimizes the Negative Evidence Lower Bound (-ELBO).
    
    ELBO = E_q [ log p(y | theta) ] - KL(q || p)
    
    Assuming a standard normal prior p(theta) = N(0, I) for simplified KL,
    or just maximizing E_q[log p] + H[q] (generalized VI / max entropy).
    
    Here we implement: Loss = - ( Expected Log Likelihood + Entropy )
    (Equivalent to minimizing KL(q || p) where p is the unnormalized posterior).
    
    Wait, usually:
    ELBO = E_q[log p(y, theta) - log q(theta)]
         = E_q[log p(y|theta)] + E_q[log p(theta)] - E_q[log q(theta)]
         = E_q[log p(y|theta)] - KL(q || p_prior)
         
    We assume p_prior is flat (improper) or implicitly handled. 
    If we want maximum likelihood estimation with uncertainty, maximizing 
    Entropy is needed to prevent collapse? No, likelihood constrains it.
    
    Let's implement:
    Loss = - ( LogLikelihood(sample) + weight * Entropy(q) )
    """
    
    tissue_model: eqx.Module
    acquisition: Any
    sigma_noise: float = 1.0

    # Define typical scales for parameters to keep optimization O(1)
    # This prevents vanishing gradients for parameters like 1e-9
    PARAM_SCALES = {
        'diameter': 1e-6,
        'diffusion_constant': 1e-9,
        'lambda_par': 1e-9,
        'lambda_perp': 1e-9
    }
    
    def log_likelihood(self, params: Dict[str, float], data: Array) -> Float[Array, ""]:
        """
        Computes log p(y | theta).
        Applies Softplus AND Scaling to ensure positivity and O(1) optimization.
        """
        safe_params = {}
        for k, v in params.items():
            if k in ['diameter', 'diffusion_constant', 'lambda_par', 'lambda_perp']:
                # theta = softplus(param) * scale
                scale = self.PARAM_SCALES.get(k, 1.0)
                val = jax.nn.softplus(v) * scale
                # Clamp to sensible physical limits to avoid numerical explosions in signal models
                # e.g. diameter < 50um, diffusivity < 3e-9 (water at 37C)
                if k == 'diameter':
                    val = jnp.clip(val, 1e-9, 50e-6)
                elif k == 'diffusion_constant':
                    val = jnp.clip(val, 1e-12, 5e-9)
                safe_params[k] = val
            else:
                safe_params[k] = v
        
        predicted = self.tissue_model(
            bvals=self.acquisition.bvalues,
            gradient_directions=self.acquisition.gradient_directions,
            big_delta=self.acquisition.Delta,
            small_delta=self.acquisition.delta,
            **safe_params
        )
        
        # Gaussian Log Likelihood
        return jnp.sum(-(data - predicted)**2 / (2 * self.sigma_noise**2))

    def loss_fn(self, vi_posterior: MeanFieldGaussian, data: Array, key: jax.Array, num_mc_samples: int = 10) -> float:
        """
        Computes the negative ELBO using Monte Carlo samples.
        """
        keys = jax.random.split(key, num_mc_samples)
        
        def single_sample_loss(k):
            # Sample parameters
            params, _ = vi_posterior.sample(k)
            # Log Likelihood
            ll = self.log_likelihood(params, data)
            return ll
            
        # Expectation of Log Likelihood
        expected_ll = jnp.mean(jax.vmap(single_sample_loss)(keys))
        
        # Entropy
        entropy = vi_posterior.entropy()
        
        return -(expected_ll + entropy)

def inverse_softplus(y):
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))

def fit_vi(
    tissue_model: eqx.Module,
    acquisition: Any,
    data: Array,
    init_params: Dict[str, float],
    sigma_noise: float = 0.02, # Typical MRI noise
    learning_rate: float = 0.01,
    num_steps: int = 1000,
    seed: int = 0
):
    """
    Fits posterior distributions q(theta) using Variational Inference.
    
    Args:
        init_params: Initial guess in PHYSICAL space (e.g. diameter in meters).
                     These are automatically transformed to unconstrained+scaled space for optimization.
    """
    key = jax.random.PRNGKey(seed)
    
    # Define scales locally to match VIMinimizer (should ideally be shared or passed)
    # Using the same dict as VIMinimizer
    PARAM_SCALES = {
        'diameter': 1e-6,
        'diffusion_constant': 1e-9,
        'lambda_par': 1e-9,
        'lambda_perp': 1e-9
    }
    
    # Transform physical initial params to unconstrained space using inverse softplus
    # Value_unconstrained = inverse_softplus(Value_physical / scale)
    
    unconstrained_init = {}
    positive_params = ['diameter', 'diffusion_constant', 'lambda_par', 'lambda_perp']
    
    for k, v in init_params.items():
        if k in positive_params:
            scale = PARAM_SCALES.get(k, 1.0)
            scaled_val = jnp.array(v) / scale
            unconstrained_init[k] = inverse_softplus(scaled_val)
        else:
            unconstrained_init[k] = jnp.array(v)
    
    # Initialize VI Posterior
    vi_posterior = MeanFieldGaussian(unconstrained_init, init_log_std=-3.0) # sigma ~ 0.05 in scaled space
    
    minimizer = VIMinimizer(tissue_model, acquisition, sigma_noise)
    
    # Optimizer (Optax for stochastic optimization)
    # Use Adabelief or Adam with reasonable LR for O(1) params
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(vi_posterior)
    
    @eqx.filter_jit
    def step(posterior, opt_state, k, y):
        loss, grads = eqx.filter_value_and_grad(minimizer.loss_fn)(posterior, y, k)
        updates, opt_state = optimizer.update(grads, opt_state, posterior)
        posterior = eqx.apply_updates(posterior, updates)
        return posterior, opt_state, loss

    # Training Loop
    keys = jax.random.split(key, num_steps)
    
    def scan_step(carrier, k):
        post, opt_st = carrier
        post, opt_st, loss = step(post, opt_st, k, data)
        return (post, opt_st), loss

    (vi_posterior, opt_state), losses = jax.lax.scan(
        scan_step, 
        (vi_posterior, opt_state), 
        keys
    )
    
    return vi_posterior, losses
