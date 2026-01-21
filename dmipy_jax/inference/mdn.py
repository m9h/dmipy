import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable

class MixtureDensityNetwork(eqx.Module):
    """
    Mixture Density Network (MDN) module.
    Predicts parameters for a Gaussian Mixture Model: mixing coefficients (pi), means (mu), and scales (sigma).
    """
    shared_layers: eqx.nn.MLP
    pi_head: eqx.nn.Linear
    mu_head: eqx.nn.Linear
    sigma_head: eqx.nn.Linear
    
    num_components: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_components: int,
        width_size: int,
        depth: int,
        key: jax.Array,
        activation: Callable = jax.nn.relu
    ):
        """
        Args:
            in_features: Number of input features.
            out_features: Number of output dimensions (per component).
            num_components: Number of Gaussian components.
            width_size: Width of hidden layers.
            depth: Number of hidden layers.
            key: PRNG key.
            activation: Activation function.
        """
        key_shared, key_pi, key_mu, key_sigma = jr.split(key, 4)
        
        self.num_components = num_components
        self.out_features = out_features
        
        self.shared_layers = eqx.nn.MLP(
            in_size=in_features,
            out_size=width_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key_shared
        )
        
        # Output heads
        # Pi: Mixing coefficients (num_components)
        self.pi_head = eqx.nn.Linear(width_size, num_components, key=key_pi)
        
        # Mu: Means (num_components * out_features)
        self.mu_head = eqx.nn.Linear(width_size, num_components * out_features, key=key_mu)
        
        # Sigma: Scales (num_components * out_features) - Diagonal assumption
        self.sigma_head = eqx.nn.Linear(width_size, num_components * out_features, key=key_sigma)

    def __call__(self, x: jax.Array):
        """
        Forward pass.
        Returns:
            logits_pi: Quantities to be softmaxed for mixing coefficients (num_components,).
            mu: Means (num_components, out_features).
            log_sigma: Log scales (num_components, out_features).
        """
        shared_out = self.shared_layers(x)
        
        logits_pi = self.pi_head(shared_out)
        
        mu_flat = self.mu_head(shared_out)
        mu = jnp.reshape(mu_flat, (self.num_components, self.out_features))
        
        log_sigma_flat = self.sigma_head(shared_out)
        log_sigma = jnp.reshape(log_sigma_flat, (self.num_components, self.out_features))
        
        return logits_pi, mu, log_sigma

def mdn_loss(model, x, y):
    """
    Computes Negative Log Likelihood (NLL) for MDN.
    
    Args:
        model: MDN instance.
        x: Input (features).
        y: Target (ground truth).
    """
    # Forward pass
    logits_pi, mu, log_sigma = model(x) # pi: (K,), mu: (K, D), sigma: (K, D)
    
    # y shape: (D,)
    # Expand y to match K components: (K, D)
    y_expanded = jnp.expand_dims(y, 0) # (1, D)
    
    # Compute Gaussian log probability for each component k
    # log N(y | mu_k, sigma_k) = -0.5 * sum((y - mu_k)^2 / sigma_k^2) - sum(log sigma_k) - 0.5 * D * log(2pi)
    # We work with log_sigma directly.
    sigma = jnp.exp(log_sigma)
    var = jnp.square(sigma)
    
    # Mahalanobis term: (y - mu)^2 / var
    diff_sq = jnp.square(y_expanded - mu)
    mahalanobis = jnp.sum(diff_sq / var, axis=-1) # (K,)
    
    # Log determinant term: sum(log_sigma)
    log_det = jnp.sum(log_sigma, axis=-1) # (K,)
    
    # Constant term
    D = y.shape[-1]
    const = 0.5 * D * jnp.log(2 * jnp.pi)
    
    # Log prob per component
    log_prob_k = -0.5 * mahalanobis - log_det - const
    
    # Combine with mixing coefficients using log-sum-exp
    # log p(y|x) = log sum_k pi_k N(y|...)
    #            = log sum_k exp(log_pi_k + log_prob_k)
    log_pi = jax.nn.log_softmax(logits_pi)
    log_prob = jax.nn.logsumexp(log_pi + log_prob_k)
    
    return -log_prob

def sample_posterior(model, x, key, n_samples=1):
    """
    Sample from the posterior distribution predicted by MDN.
    """
    key_cat, key_gauss = jr.split(key)
    
    logits_pi, mu, log_sigma = model(x)
    sigma = jnp.exp(log_sigma)
    
    # Select component
    # logits_pi is (K,)
    k_indices = jr.categorical(key_cat, logits_pi, shape=(n_samples,))
    
    # Gather component params
    # mu: (K, D), sigma: (K, D)
    mu_sel = mu[k_indices] # (n_samples, D)
    sigma_sel = sigma[k_indices] # (n_samples, D)
    
    # Sample from Gaussian
    epsilon = jr.normal(key_gauss, shape=mu_sel.shape)
    samples = mu_sel + sigma_sel * epsilon
    
    return samples
