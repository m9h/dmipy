"""SBI (Simulation-Based Inference) for qMRI parameter estimation.

Trains a neural posterior estimator on simulated qMRI data, then performs
amortized inference: one forward pass per voxel instead of iterative
optimization. This solves the local minima problem in mcDESPOT and the
speed problem for fitting 170 WAND subjects.

Architecture:
  1. Define prior over parameters (physiological ranges)
  2. Simulate training data: sample params → forward model → add noise
  3. Train MDN/flow to approximate p(params | data)
  4. At inference: data → neural network → posterior (mean + uncertainty)

Uses sbi4dwi's existing infrastructure:
  - inference/mdn.py: Mixture Density Networks
  - inference/flows.py: Normalizing Flows
  - inference/amortized.py: Amortized inference framework
"""

from __future__ import annotations
from typing import Callable, NamedTuple, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


class SBIResult(NamedTuple):
    """Result from SBI inference with full posterior."""
    mean: jnp.ndarray      # (n_params,) posterior mean
    std: jnp.ndarray       # (n_params,) posterior std
    samples: jnp.ndarray   # (n_samples, n_params) posterior samples
    log_prob: float         # log probability of the MAP estimate


# ---------------------------------------------------------------------------
# Simulator: generates (params, data) training pairs
# ---------------------------------------------------------------------------

def build_simulator(
    forward_fn: Callable,
    prior_low: jnp.ndarray,
    prior_high: jnp.ndarray,
    noise_std: float = 0.02,
):
    """Build a simulator that generates (params, data) pairs from the prior.

    Parameters
    ----------
    forward_fn : callable(params) → signal
        The qMRI forward model (e.g., mcdespot_combined_signal).
    prior_low : (n_params,) — lower bounds of uniform prior.
    prior_high : (n_params,) — upper bounds of uniform prior.
    noise_std : float — relative noise standard deviation.
    """
    n_params = prior_low.shape[0]

    def simulate(key):
        k1, k2 = jr.split(key)
        # Sample from uniform prior
        params = jr.uniform(k1, (n_params,), minval=prior_low, maxval=prior_high)
        # Forward model
        signal = forward_fn(params)
        # Add Rician-like noise (Gaussian approx for high SNR)
        noise = jr.normal(k2, signal.shape) * noise_std * jnp.mean(jnp.abs(signal))
        data = signal + noise
        return params, data

    return simulate


def generate_training_data(
    simulator: Callable,
    n_samples: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate training dataset of (params, data) pairs.

    Returns
    -------
    params : (n_samples, n_params)
    data : (n_samples, n_data_points)
    """
    keys = jr.split(key, n_samples)
    params, data = jax.vmap(simulator)(keys)
    return params, data


# ---------------------------------------------------------------------------
# Simple MDN (Mixture Density Network) for posterior estimation
# ---------------------------------------------------------------------------

class SimpleMDN(eqx.Module):
    """Lightweight Mixture Density Network for qMRI posterior estimation.

    Maps observed data → mixture of Gaussians over parameters.
    """
    layers: list
    mu_head: eqx.nn.Linear
    log_sigma_head: eqx.nn.Linear
    log_pi_head: eqx.nn.Linear
    n_components: int = eqx.field(static=True)
    n_params: int = eqx.field(static=True)

    def __init__(self, n_data: int, n_params: int, n_components: int = 5,
                 hidden_size: int = 64, *, key):
        k1, k2, k3, k4, k5 = jr.split(key, 5)
        self.n_components = n_components
        self.n_params = n_params
        self.layers = [
            eqx.nn.Linear(n_data, hidden_size, key=k1),
            eqx.nn.Linear(hidden_size, hidden_size, key=k2),
            eqx.nn.Linear(hidden_size, hidden_size, key=k3),
        ]
        self.mu_head = eqx.nn.Linear(hidden_size, n_components * n_params, key=k4)
        self.log_sigma_head = eqx.nn.Linear(hidden_size, n_components * n_params, key=k5)
        self.log_pi_head = eqx.nn.Linear(hidden_size, n_components, key=jr.fold_in(key, 99))

    def __call__(self, data: jnp.ndarray):
        """Forward pass: data → mixture parameters."""
        x = data
        for layer in self.layers:
            x = jax.nn.gelu(layer(x))

        K, P = self.n_components, self.n_params
        mu = self.mu_head(x).reshape(K, P)
        log_sigma = self.log_sigma_head(x).reshape(K, P)
        sigma = jax.nn.softplus(log_sigma) + 1e-4
        log_pi = self.log_pi_head(x)
        pi = jax.nn.softmax(log_pi)

        return mu, sigma, pi

    def posterior_mean_std(self, data: jnp.ndarray):
        """Compute posterior mean and std from the mixture."""
        mu, sigma, pi = self(data)
        # Mixture mean: E[x] = sum_k pi_k * mu_k
        mean = jnp.sum(pi[:, None] * mu, axis=0)
        # Mixture variance: Var[x] = sum_k pi_k * (sigma_k^2 + mu_k^2) - mean^2
        var = jnp.sum(pi[:, None] * (sigma ** 2 + mu ** 2), axis=0) - mean ** 2
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        return mean, std

    def sample(self, data: jnp.ndarray, key: jax.Array, n_samples: int = 100):
        """Sample from the posterior."""
        mu, sigma, pi = self(data)
        k1, k2 = jr.split(key)
        # Sample component indices
        component_idx = jr.categorical(k1, jnp.log(pi + 1e-10), shape=(n_samples,))
        # Sample from selected Gaussians
        noise = jr.normal(k2, (n_samples, self.n_params))
        samples = mu[component_idx] + sigma[component_idx] * noise
        return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sbi(
    mdn: SimpleMDN,
    train_params: jnp.ndarray,
    train_data: jnp.ndarray,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    key: Optional[jax.Array] = None,
) -> tuple[SimpleMDN, list[float]]:
    """Train the MDN on simulated (params, data) pairs.

    Loss: negative log-likelihood of the mixture model.
    """
    if key is None:
        key = jr.PRNGKey(0)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(mdn, eqx.is_array))

    N = train_params.shape[0]
    history = []

    def nll_loss(mdn, data_batch, params_batch):
        """Negative log-likelihood of params under the MDN posterior."""
        def single_nll(data, params):
            mu, sigma, pi = mdn(data)
            # Log-likelihood of params under each component
            log_probs = -0.5 * jnp.sum(((params[None, :] - mu) / sigma) ** 2, axis=1) \
                        - jnp.sum(jnp.log(sigma), axis=1) \
                        + jnp.log(pi + 1e-10)
            return -jax.scipy.special.logsumexp(log_probs)

        return jnp.mean(jax.vmap(single_nll)(data_batch, params_batch))

    @eqx.filter_jit
    def train_step(mdn, opt_state, data_batch, params_batch):
        loss, grads = eqx.filter_value_and_grad(nll_loss)(mdn, data_batch, params_batch)
        updates, opt_state = optimizer.update(grads, opt_state, mdn)
        mdn = eqx.apply_updates(mdn, updates)
        return mdn, opt_state, loss

    for epoch in range(n_epochs):
        key, k_perm = jr.split(key)
        perm = jr.permutation(k_perm, N)
        epoch_loss = 0.0
        n_batches = max(1, N // batch_size)

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            mdn, opt_state, loss = train_step(
                mdn, opt_state, train_data[idx], train_params[idx]
            )
            epoch_loss += float(loss)

        history.append(epoch_loss / n_batches)

    return mdn, history


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def sbi_inference(
    mdn: SimpleMDN,
    data: jnp.ndarray,
    key: jax.Array,
    n_samples: int = 1000,
) -> SBIResult:
    """Run amortized inference: data → posterior.

    Parameters
    ----------
    mdn : trained MDN.
    data : (n_data_points,) observed signal.
    key : PRNG key for sampling.
    n_samples : number of posterior samples.

    Returns
    -------
    SBIResult with mean, std, samples, log_prob.
    """
    mean, std = mdn.posterior_mean_std(data)
    samples = mdn.sample(data, key, n_samples)

    # Log probability at the mean
    mu, sigma, pi = mdn(data)
    log_probs = -0.5 * jnp.sum(((mean[None, :] - mu) / sigma) ** 2, axis=1) \
                - jnp.sum(jnp.log(sigma), axis=1) + jnp.log(pi + 1e-10)
    log_prob = float(jax.scipy.special.logsumexp(log_probs))

    return SBIResult(mean=mean, std=std, samples=samples, log_prob=log_prob)
