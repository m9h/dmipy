"""
Score-based diffusion posterior estimator for microstructure parameters.

Uses an E(3)-equivariant score network to learn the posterior
∇_θ log p(θ | x), where θ contains scalar parameters (0e irreps)
and orientation unit vectors (1o irreps).

The key advantage over normalizing flows: no invertibility constraint,
naturally handles multimodal posteriors, and E(3) equivariance ensures
orientations transform correctly by construction.

References
----------
- Song et al., "Score-Based Generative Modeling through SDEs" (2021)
- Geffner et al., "Compositional Score Modeling for SBI" (2023)
"""

from __future__ import annotations

import time
from typing import Tuple, Optional, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import e3nn_jax as e3nn
from jaxtyping import Array, Float, PRNGKeyArray


# ------------------------------------------------------------------ #
# Variance-preserving SDE noise schedule
# ------------------------------------------------------------------ #

class VPSchedule(eqx.Module):
    """Variance-preserving noise schedule for the forward diffusion."""
    beta_min: float
    beta_max: float

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def alpha_bar(self, t: Float[Array, ""]) -> Float[Array, ""]:
        """Cumulative signal retention at time *t* ∈ [0, 1]."""
        log_ab = -0.5 * (self.beta_min * t
                         + 0.5 * (self.beta_max - self.beta_min) * t ** 2)
        return jnp.exp(log_ab)

    def noise_and_signal(self, t):
        """Return ``(signal_rate, noise_rate)``."""
        ab = self.alpha_bar(t)
        return jnp.sqrt(ab), jnp.sqrt(1.0 - ab)

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)


# ------------------------------------------------------------------ #
# Equivariant building blocks
# ------------------------------------------------------------------ #

class E3Linear(eqx.Module):
    """Equinox wrapper around ``e3nn.FunctionalLinear``."""
    weight: jax.Array
    linear: e3nn.FunctionalLinear

    def __init__(self, irreps_in, irreps_out, key):
        self.linear = e3nn.FunctionalLinear(irreps_in, irreps_out)
        self.weight = jax.random.normal(key, (self.linear.num_weights,)) * 0.02

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return self.linear(self.weight, x)


class FiLMModulation(eqx.Module):
    """FiLM conditioning: produces per-channel scale and shift for 0e irreps."""
    gamma: eqx.nn.Linear
    beta: eqx.nn.Linear

    def __init__(self, cond_dim: int, n_scalars: int, key):
        k1, k2 = jax.random.split(key)
        self.gamma = eqx.nn.Linear(cond_dim, n_scalars, key=k1)
        self.beta = eqx.nn.Linear(cond_dim, n_scalars, key=k2)

    def __call__(self, cond):
        return self.gamma(cond), self.beta(cond)


# ------------------------------------------------------------------ #
# Score network
# ------------------------------------------------------------------ #

class MicrostructureScoreNet(eqx.Module):
    """E(3)-equivariant score network for microstructure posterior estimation.

    The parameter vector θ is represented as ``n_scalars × 0e + n_vectors × 1o``
    irreps.  Scalar parameters (diffusivities, fractions) are 0e; orientation
    unit vectors are 1o.  The score output has the same irreps, guaranteeing
    equivariance: ``score(Rθ, t, x) = R · score(θ, t, x)``.

    Conditioning (dMRI signal + diffusion time) is injected via FiLM
    modulation on scalar channels, which preserves equivariance.
    """

    layers: list
    film_layers: list
    signal_encoder: eqx.nn.MLP
    time_encoder: eqx.nn.MLP
    param_irreps: str
    hidden_irreps: str
    n_hidden_scalars: int

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        n_scalars: int = 4,
        n_vectors: int = 2,
        signal_dim: int = 90,
        hidden_scalars: int = 64,
        hidden_vectors: int = 21,
        depth: int = 4,
        cond_dim: int = 128,
    ):
        self.param_irreps = f"{n_scalars}x0e + {n_vectors}x1o"
        self.n_hidden_scalars = hidden_scalars
        self.hidden_irreps = f"{hidden_scalars}x0e + {hidden_vectors}x1o"

        keys = jax.random.split(key, 2 * depth + 4)

        # Conditioning encoders
        self.signal_encoder = eqx.nn.MLP(
            in_size=signal_dim, out_size=cond_dim, width_size=cond_dim,
            depth=2, activation=jax.nn.gelu, key=keys[0],
        )
        self.time_encoder = eqx.nn.MLP(
            in_size=1, out_size=cond_dim, width_size=cond_dim,
            depth=2, activation=jax.nn.silu, key=keys[1],
        )

        # E3 layers + FiLM conditioners
        self.layers = []
        self.film_layers = []

        # Input projection
        self.layers.append(E3Linear(self.param_irreps, self.hidden_irreps, keys[2]))
        self.film_layers.append(FiLMModulation(cond_dim * 2, hidden_scalars, keys[3]))

        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(E3Linear(self.hidden_irreps, self.hidden_irreps, keys[4 + 2 * i]))
            self.film_layers.append(FiLMModulation(cond_dim * 2, hidden_scalars, keys[5 + 2 * i]))

        # Output projection (no FiLM, no activation)
        self.layers.append(E3Linear(self.hidden_irreps, self.param_irreps, keys[-1]))

    def __call__(
        self,
        theta_t: Float[Array, "param_dim"],
        t: Float[Array, ""],
        signal: Float[Array, "signal_dim"],
    ) -> Float[Array, "param_dim"]:
        """Predict noise ε given noisy parameters, time, and signal."""
        # Conditioning embedding
        sig_emb = self.signal_encoder(signal)
        t_emb = self.time_encoder(jnp.atleast_1d(t))
        cond = jnp.concatenate([sig_emb, t_emb])

        # Wrap as IrrepsArray
        x = e3nn.IrrepsArray(self.param_irreps, theta_t)

        # Forward with FiLM conditioning
        for layer, film in zip(self.layers[:-1], self.film_layers):
            x = layer(x)
            x = self._film_and_activate(x, cond, film)

        # Output projection
        x = self.layers[-1](x)
        return x.array

    def _film_and_activate(self, x, cond, film):
        gamma, beta = film(cond)
        arr = x.array

        # FiLM on scalar (0e) channels only — preserves equivariance
        ns = self.n_hidden_scalars
        scalars = arr[:ns] * (1.0 + gamma) + beta
        vectors = arr[ns:]
        arr = jnp.concatenate([scalars, vectors])
        x = e3nn.IrrepsArray(x.irreps, arr)

        # Equivariant norm activation
        n_irrep_groups = len(e3nn.Irreps(x.irreps))
        acts = [jax.nn.gelu] * n_irrep_groups
        x = e3nn.norm_activation(x, acts)
        return x


# ------------------------------------------------------------------ #
# MLP score network (no equivariance — plain residual MLP)
# ------------------------------------------------------------------ #

class MLPScoreNet(eqx.Module):
    """Plain MLP score network with residual connections and FiLM conditioning.

    This serves as the baseline to prove the denoising score matching
    objective works before adding equivariance back in.
    """
    layers: list
    signal_encoder: eqx.nn.MLP
    time_encoder: eqx.nn.MLP
    cond_proj: list  # per-layer FiLM: (gamma_linear, beta_linear)
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        param_dim: int = 10,
        signal_dim: int = 90,
        hidden_dim: int = 256,
        depth: int = 6,
        cond_dim: int = 128,
    ):
        keys = jax.random.split(key, depth + 5)

        # Conditioning encoders
        self.signal_encoder = eqx.nn.MLP(
            in_size=signal_dim, out_size=cond_dim, width_size=cond_dim,
            depth=2, activation=jax.nn.gelu, key=keys[0],
        )
        self.time_encoder = eqx.nn.MLP(
            in_size=1, out_size=cond_dim, width_size=cond_dim // 2,
            depth=2, activation=jax.nn.silu, key=keys[1],
        )

        # Input projection: param_dim → hidden_dim
        self.layers = [eqx.nn.Linear(param_dim, hidden_dim, key=keys[2])]
        self.cond_proj = []

        # Hidden layers with FiLM conditioning
        for i in range(depth - 1):
            ki = keys[3 + i]
            k_l, k_g, k_b = jax.random.split(ki, 3)
            self.layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=k_l))
            self.cond_proj.append((
                eqx.nn.Linear(cond_dim * 2, hidden_dim, key=k_g),
                eqx.nn.Linear(cond_dim * 2, hidden_dim, key=k_b),
            ))

        # Output projection
        self.output_layer = eqx.nn.Linear(hidden_dim, param_dim, key=keys[-1])

    def __call__(
        self,
        theta_t: Float[Array, "param_dim"],
        t: Float[Array, ""],
        signal: Float[Array, "signal_dim"],
    ) -> Float[Array, "param_dim"]:
        """Predict noise ε."""
        sig_emb = self.signal_encoder(signal)
        t_emb = self.time_encoder(jnp.atleast_1d(t))
        cond = jnp.concatenate([sig_emb, t_emb])

        # Input projection
        h = self.layers[0](theta_t)
        h = jax.nn.gelu(h)

        # Residual blocks with FiLM
        for layer, (gamma_proj, beta_proj) in zip(self.layers[1:], self.cond_proj):
            h_in = h
            gamma = gamma_proj(cond)
            beta = beta_proj(cond)
            h = layer(h)
            h = h * (1.0 + gamma) + beta  # FiLM
            h = jax.nn.gelu(h)
            h = h + h_in  # residual

        return self.output_layer(h)


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #

def train_score_posterior(
    key: PRNGKeyArray,
    score_net: MicrostructureScoreNet,
    *,
    simulator_fn: Callable,
    prior_fn: Callable,
    schedule: VPSchedule,
    num_steps: int = 50_000,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    print_every: int = 1000,
):
    """Train the score network via denoising score matching.

    Parameters
    ----------
    score_net : MicrostructureScoreNet
        Untrained score network.
    simulator_fn : callable
        ``(key, theta) -> signal``  — forward model + noise + b0-norm.
    prior_fn : callable
        ``(key, n) -> theta``  — prior sampler.
    schedule : VPSchedule
        Noise schedule.

    Returns
    -------
    score_net : MicrostructureScoreNet
        Trained network.
    losses : list of float
    """
    import optax

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(score_net, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, opt_state, theta, signal, t, eps):
        """Single denoising score matching step."""
        def loss_fn(net):
            sig_rate, noise_rate = schedule.noise_and_signal(t)
            # Noisy parameters: θ_t = α(t)·θ + σ(t)·ε
            theta_t = sig_rate[:, None] * theta + noise_rate[:, None] * eps
            # Predict noise
            eps_pred = jax.vmap(net)(theta_t, t, signal)
            return jnp.mean((eps_pred - eps) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        grads_f = eqx.filter(grads, eqx.is_inexact_array)
        params_f = eqx.filter(net, eqx.is_inexact_array)
        updates, opt_state_new = optimizer.update(grads_f, opt_state, params_f)
        net = eqx.apply_updates(net, updates)
        return net, opt_state_new, loss

    curr_key = key
    losses = []
    t0 = time.time()

    for i in range(num_steps):
        curr_key, k_step = jax.random.split(curr_key)
        k1, k2, k3, k4 = jax.random.split(k_step, 4)

        # Sample training batch
        theta = prior_fn(k1, batch_size)
        signal = simulator_fn(k2, theta)

        # Sample diffusion time and noise
        t = jax.random.uniform(k3, (batch_size,), minval=1e-5, maxval=1.0)
        eps = jax.random.normal(k4, theta.shape)

        score_net, opt_state, loss = step(score_net, opt_state, theta, signal, t, eps)
        losses.append(float(loss))

        if i % print_every == 0:
            print(f"[Score] step {i}/{num_steps}  loss={loss:.4f}")

    elapsed = time.time() - t0
    print(f"[Score] Training done. {num_steps} steps in {elapsed:.1f}s "
          f"({num_steps / elapsed:.0f} steps/s)")
    return score_net, losses


# ------------------------------------------------------------------ #
# Sampling (reverse SDE — Euler-Maruyama)
# ------------------------------------------------------------------ #

def sample_posterior(
    key: PRNGKeyArray,
    score_net: MicrostructureScoreNet,
    signal: Float[Array, "signal_dim"],
    schedule: VPSchedule,
    *,
    n_samples: int = 500,
    n_steps: int = 200,
    n_scalars: int = 4,
    n_vectors: int = 2,
) -> Float[Array, "n_samples param_dim"]:
    """Draw posterior samples via reverse-SDE integration.

    Parameters
    ----------
    signal : array (signal_dim,)
        Conditioning dMRI signal for a single voxel.
    n_samples : int
        Number of posterior samples to draw.
    n_steps : int
        Number of reverse-SDE discretisation steps.

    Returns
    -------
    samples : array (n_samples, param_dim)
    """
    param_dim = n_scalars + n_vectors * 3
    dt = 1.0 / n_steps

    def reverse_step(carry, step_idx):
        theta_t, key = carry
        t = 1.0 - step_idx * dt

        # Score = -eps_pred / noise_rate
        _, noise_rate = schedule.noise_and_signal(t)
        eps_pred = jax.vmap(score_net, in_axes=(0, None, None))(
            theta_t, t, signal
        )
        score = -eps_pred / jnp.maximum(noise_rate, 1e-6)

        # Reverse SDE: dθ = [f(θ,t) - g²(t)·score] dt + g(t) dW
        beta_t = schedule.beta(t)
        drift = -0.5 * beta_t * theta_t - beta_t * score
        diffusion = jnp.sqrt(beta_t)

        key, k_noise = jax.random.split(key)
        noise = jax.random.normal(k_noise, theta_t.shape)

        # Euler-Maruyama update
        theta_t = theta_t + drift * dt + diffusion * jnp.sqrt(dt) * noise
        return (theta_t, key), None

    # Start from Gaussian noise
    k_init, k_loop = jax.random.split(key)
    theta_T = jax.random.normal(k_init, (n_samples, param_dim))

    step_indices = jnp.arange(n_steps)
    (theta_0, _), _ = jax.lax.scan(reverse_step, (theta_T, k_loop), step_indices)

    # Normalise orientation vectors back to unit sphere
    for v in range(n_vectors):
        start = n_scalars + v * 3
        end = start + 3
        vec = theta_0[:, start:end]
        vec = vec / jnp.maximum(jnp.linalg.norm(vec, axis=-1, keepdims=True), 1e-8)
        theta_0 = theta_0.at[:, start:end].set(vec)

    return theta_0
