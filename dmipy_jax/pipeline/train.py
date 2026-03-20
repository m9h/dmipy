"""
Unified training entry-point for MDN and flow-based SBI.
"""

from __future__ import annotations

import time
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.inference.mdn import MixtureDensityNetwork, mdn_loss


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _get_activation(name: str):
    """Return a JAX activation function by name."""
    return {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "silu": jax.nn.silu}.get(
        name, jax.nn.gelu
    )


def _build_lr_schedule(config: SBIPipelineConfig):
    """Return an optax learning-rate schedule (or scalar) from config."""
    if config.lr_schedule == "constant":
        return config.learning_rate
    elif config.lr_schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=config.n_steps
        )
    elif config.lr_schedule == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.n_steps,
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule!r}")


def train_sbi(
    config: SBIPipelineConfig,
    simulator: ModelSimulator,
    *,
    key: Optional[jax.Array] = None,
    print_every: int = 500,
    trackio_run: bool = False,
):
    """Train an SBI model (MDN or flow) and return the trained network.

    Parameters
    ----------
    config : SBIPipelineConfig
        Full pipeline configuration.
    simulator : ModelSimulator
        Simulator that provides ``sample_and_simulate``.
    key : jax.Array, optional
        PRNG key; defaults to ``config.seed``.
    print_every : int
        Logging interval.
    trackio_run : bool
        If *True*, log metrics to an already-initialised Trackio run.

    Returns
    -------
    model
        Trained Equinox module (MDN or flow).
    losses : list of float
        Per-step training losses.
    """
    if key is None:
        key = jax.random.key(config.seed)

    if config.inference_mode == "mdn":
        return _train_mdn(config, simulator, key, print_every)
    elif config.inference_mode == "flow":
        return _train_flow(config, simulator, key, print_every, trackio_run)
    elif config.inference_mode == "score":
        return _train_score(config, simulator, key, print_every)
    else:
        raise ValueError(f"Unknown inference_mode: {config.inference_mode!r}")


# --------------------------------------------------------------------------- #
# MDN training
# --------------------------------------------------------------------------- #

def _train_mdn(config, simulator, key, print_every):
    k_net, k_train = jax.random.split(key)

    # Compute normalisation bounds from parameter ranges
    lows = jnp.array([simulator.parameter_ranges[n][0]
                       for n in simulator.parameter_names])
    highs = jnp.array([simulator.parameter_ranges[n][1]
                        for n in simulator.parameter_names])
    spans = jnp.maximum(highs - lows, 1e-12)

    if config.architecture == "residual":
        from dmipy_jax.inference.mdn import ResidualMDN
        model = ResidualMDN(
            in_features=simulator.signal_dim,
            out_features=config.theta_dim,
            num_components=config.n_components,
            width_size=config.hidden_dim,
            depth=config.depth,
            key=k_net,
            activation=_get_activation(config.activation),
        )
    else:
        model = MixtureDensityNetwork(
            in_features=simulator.signal_dim,
            out_features=config.theta_dim,
            num_components=config.n_components,
            width_size=config.hidden_dim,
            depth=config.depth,
            key=k_net,
            activation=_get_activation(config.activation),
        )

    # LR schedule
    lr_schedule = _build_lr_schedule(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # EMA initialisation — deep copy of initial model
    if config.use_ema:
        ema_model = jax.tree.map(lambda x: x, model)

    # Batch loss: vmap the per-sample MDN NLL
    def batch_loss(mdl, x_batch, y_batch):
        per_sample = jax.vmap(mdn_loss, in_axes=(None, 0, 0))(mdl, x_batch, y_batch)
        return jnp.mean(per_sample)

    @eqx.filter_jit
    def step(mdl, opt_st, x, y_norm):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(mdl, x, y_norm)
        updates, opt_st = optimizer.update(grads, opt_st, mdl)
        mdl = eqx.apply_updates(mdl, updates)
        return mdl, opt_st, loss

    @eqx.filter_jit
    def compute_val_loss(mdl, x, y_norm):
        return batch_loss(mdl, x, y_norm)

    losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_step = 0
    steps_without_improvement = 0
    curr_key = k_train
    t0 = time.time()

    for i in range(config.n_steps):
        if config.curriculum_noise and simulator.snr_range is not None:
            simulator.curriculum_step = (i, config.n_steps)
        curr_key, k_step = jax.random.split(curr_key)
        theta, signals = simulator.sample_and_simulate(k_step, config.batch_size)
        # Normalise targets to [0, 1]
        theta_norm = (theta - lows) / spans

        # Split batch into train / validation portions
        n_total = theta_norm.shape[0]
        n_val = max(1, int(n_total * config.val_fraction))
        n_train = n_total - n_val

        x_train, x_val = signals[:n_train], signals[n_train:]
        y_train, y_val = theta_norm[:n_train], theta_norm[n_train:]

        model, opt_state, train_loss = step(model, opt_state, x_train, y_train)
        val_loss = compute_val_loss(model, x_val, y_val)

        losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        # EMA update — only average array leaves, keep static fields as-is
        if config.use_ema:
            decay = config.ema_decay
            ema_model = jax.tree.map(
                lambda e, m: decay * e + (1 - decay) * m
                if eqx.is_array(e)
                else m,
                ema_model,
                model,
            )

        # Early stopping tracking
        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            best_val_step = i
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if print_every and i % print_every == 0:
            # Resolve current learning rate for logging
            if callable(lr_schedule):
                current_lr = float(lr_schedule(i))
            else:
                current_lr = float(lr_schedule)
            print(
                f"[MDN] step {i}/{config.n_steps}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={current_lr:.2e}"
            )

        if config.patience > 0 and steps_without_improvement >= config.patience:
            print(
                f"[MDN] Early stopping at step {i}: no val improvement "
                f"for {config.patience} steps (best={best_val_loss:.4f} "
                f"at step {best_val_step})"
            )
            break

    elapsed = time.time() - t0
    actual_steps = len(losses)
    print(f"[MDN] Training done. {actual_steps} steps in {elapsed:.1f}s "
          f"({actual_steps / elapsed:.0f} steps/s)")

    # Use EMA model if enabled, otherwise latest model
    final_model = ema_model if config.use_ema else model

    # Store normalisation info on the model for downstream use
    final_model = _NormalisedMDN(final_model, lows, spans)
    return final_model, losses


class _NormalisedMDN(eqx.Module):
    """Wrapper that denormalises MDN outputs back to physical units."""
    inner: eqx.Module
    lows: jax.Array
    spans: jax.Array

    def __call__(self, x):
        logits_pi, mu_norm, log_sigma = self.inner(x)
        # Denormalise means back to physical space
        mu = mu_norm * self.spans + self.lows
        # Scale log_sigma accordingly: sigma_phys = sigma_norm * span
        log_sigma = log_sigma + jnp.log(self.spans)
        return logits_pi, mu, log_sigma


# --------------------------------------------------------------------------- #
# Flow (NPE) training
# --------------------------------------------------------------------------- #

def _train_flow(config, simulator, key, print_every, trackio_run=False):
    from dmipy_jax.inference.trainer import create_trainer, train_loop

    k_flow, k_train = jax.random.split(key)

    # Compute normalisation bounds
    lows = jnp.array([simulator.parameter_ranges[n][0]
                       for n in simulator.parameter_names])
    highs = jnp.array([simulator.parameter_ranges[n][1]
                        for n in simulator.parameter_names])
    spans = jnp.maximum(highs - lows, 1e-12)

    # b0 mask for normalisation
    b0_mask = simulator.acquisition.bvalues < 100e6

    # Wrap simulator to normalise theta to [0, 1] and apply noise + b0-norm
    # internally so training sees the same distribution as test/deploy.
    def sim_fn(k, theta_norm):
        k1, k2 = jax.random.split(k)
        theta = theta_norm * spans + lows
        signal = simulator.simulate(k1, theta)
        noisy = simulator.add_noise(k2, signal)
        # b0-normalise (mirrors sample_and_simulate / deploy preprocessing)
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean
        return noisy

    def prior_fn(k, n):
        theta = simulator.prior_sampler(k, n)
        return (theta - lows) / spans

    # Build the LR schedule and pass it through to create_trainer
    lr_schedule = _build_lr_schedule(config)

    flow, optimizer = create_trainer(
        flow_key=k_flow,
        theta_dim=config.theta_dim,
        signal_dim=simulator.signal_dim,
        simulator=sim_fn,
        prior_sampler=prior_fn,
        learning_rate=lr_schedule,
        hidden_dim=config.hidden_dim,
        num_layers=config.depth,
        flow_type=config.flow_type,
        knots=config.knots,
    )

    # Build a curriculum callback if enabled
    _step_callback = None
    if config.curriculum_noise and simulator.snr_range is not None:
        def _step_callback(i, total):
            simulator.curriculum_step = (i, total)

    # noise_std=0.0 because sim_fn already applies noise internally
    trained_flow, losses = train_loop(
        flow,
        optimizer,
        simulator=sim_fn,
        prior_sampler=prior_fn,
        key=k_train,
        num_steps=config.n_steps,
        batch_size=config.batch_size,
        noise_std=0.0,
        print_every=print_every,
        step_callback=_step_callback,
        trackio_run=trackio_run,
    )

    # Wrap the flow to denormalise samples back to physical space
    model = _NormalisedFlow(trained_flow, lows, spans)
    return model, losses


class _NormalisedFlow(eqx.Module):
    """Wrapper that denormalises flow samples back to physical units."""
    inner: eqx.Module
    lows: jax.Array
    spans: jax.Array

    def log_prob(self, theta, condition=None):
        theta_norm = (theta - self.lows) / self.spans
        return self.inner.log_prob(theta_norm, condition=condition)

    def sample(self, key, sample_shape=(), condition=None):
        samples_norm = self.inner.sample(key, sample_shape, condition=condition)
        return samples_norm * self.spans + self.lows


# --------------------------------------------------------------------------- #
# Score-based diffusion posterior training
# --------------------------------------------------------------------------- #

def _train_score(config, simulator, key, print_every):
    from dmipy_jax.inference.score_posterior import (
        MicrostructureScoreNet, VPSchedule, train_score_posterior,
    )

    k_net, k_train = jax.random.split(key)

    # Compute normalisation bounds
    lows = jnp.array([simulator.parameter_ranges[n][0]
                       for n in simulator.parameter_names])
    highs = jnp.array([simulator.parameter_ranges[n][1]
                        for n in simulator.parameter_names])
    spans = jnp.maximum(highs - lows, 1e-12)

    # b0 mask for normalisation
    b0_mask = simulator.acquisition.bvalues < 100e6

    # Count scalar vs vector parameters (vectors are xyz triplets)
    # Convention: parameters ending in x,y,z that come in triplets are vectors
    names = simulator.parameter_names
    n_vectors = 0
    i = 0
    while i < len(names):
        if (i + 2 < len(names)
                and names[i].endswith("x")
                and names[i + 1].endswith("y")
                and names[i + 2].endswith("z")):
            n_vectors += 1
            i += 3
        else:
            i += 1
    n_scalars = len(names) - n_vectors * 3

    # Simulator wrapper: normalises theta and applies noise + b0-norm
    def sim_fn(k, theta_norm):
        k1, k2 = jax.random.split(k)
        theta = theta_norm * spans + lows
        signal = simulator.simulate(k1, theta)
        noisy = simulator.add_noise(k2, signal)
        if jnp.any(b0_mask):
            b0_mean = jnp.mean(noisy[:, b0_mask], axis=1, keepdims=True)
            b0_mean = jnp.maximum(b0_mean, 1e-6)
            noisy = noisy / b0_mean
        return noisy

    def prior_fn(k, n):
        theta = simulator.prior_sampler(k, n)
        return (theta - lows) / spans

    schedule = VPSchedule()
    score_net = MicrostructureScoreNet(
        k_net,
        n_scalars=n_scalars,
        n_vectors=n_vectors,
        signal_dim=simulator.signal_dim,
        hidden_scalars=config.hidden_dim,
        hidden_vectors=config.hidden_dim // 3,
        depth=config.depth,
    )

    trained_net, losses = train_score_posterior(
        k_train,
        score_net,
        simulator_fn=sim_fn,
        prior_fn=prior_fn,
        schedule=schedule,
        num_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        print_every=print_every,
    )

    model = _NormalisedScore(trained_net, schedule, lows, spans,
                             n_scalars, n_vectors)
    return model, losses


class _NormalisedScore(eqx.Module):
    """Wrapper that denormalises score-based posterior samples."""
    inner: eqx.Module
    schedule: eqx.Module
    lows: jax.Array
    spans: jax.Array
    n_scalars: int
    n_vectors: int

    def sample(self, key, sample_shape=(), condition=None):
        from dmipy_jax.inference.score_posterior import sample_posterior
        n_samples = sample_shape[0] if sample_shape else 1
        samples_norm = sample_posterior(
            key, self.inner, condition, self.schedule,
            n_samples=n_samples,
            n_scalars=self.n_scalars,
            n_vectors=self.n_vectors,
        )
        return samples_norm * self.spans + self.lows
