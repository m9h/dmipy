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


def train_sbi(
    config: SBIPipelineConfig,
    simulator: ModelSimulator,
    *,
    key: Optional[jax.Array] = None,
    print_every: int = 500,
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

    Returns
    -------
    model
        Trained Equinox module (MDN or flow).
    losses : list of float
        Per-step training losses.
    """
    if key is None:
        key = jax.random.PRNGKey(config.seed)

    if config.inference_mode == "mdn":
        return _train_mdn(config, simulator, key, print_every)
    elif config.inference_mode == "flow":
        return _train_flow(config, simulator, key, print_every)
    else:
        raise ValueError(f"Unknown inference_mode: {config.inference_mode!r}")


# --------------------------------------------------------------------------- #
# MDN training
# --------------------------------------------------------------------------- #

def _train_mdn(config, simulator, key, print_every):
    k_net, k_train = jax.random.split(key)

    model = MixtureDensityNetwork(
        in_features=simulator.signal_dim,
        out_features=config.theta_dim,
        num_components=config.n_components,
        width_size=config.hidden_dim,
        depth=config.depth,
        key=k_net,
    )

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Batch loss: vmap the per-sample MDN NLL
    def batch_loss(mdl, x_batch, y_batch):
        per_sample = jax.vmap(mdn_loss, in_axes=(None, 0, 0))(mdl, x_batch, y_batch)
        return jnp.mean(per_sample)

    @eqx.filter_jit
    def step(mdl, opt_st, x, y):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(mdl, x, y)
        updates, opt_st = optimizer.update(grads, opt_st, mdl)
        mdl = eqx.apply_updates(mdl, updates)
        return mdl, opt_st, loss

    losses = []
    curr_key = k_train
    t0 = time.time()

    for i in range(config.n_steps):
        curr_key, k_step = jax.random.split(curr_key)
        theta, signals = simulator.sample_and_simulate(k_step, config.batch_size)
        model, opt_state, loss = step(model, opt_state, signals, theta)
        losses.append(float(loss))
        if print_every and i % print_every == 0:
            print(f"[MDN] step {i}/{config.n_steps}  loss={loss:.4f}")

    elapsed = time.time() - t0
    print(f"[MDN] Training done. {config.n_steps} steps in {elapsed:.1f}s "
          f"({config.n_steps / elapsed:.0f} steps/s)")
    return model, losses


# --------------------------------------------------------------------------- #
# Flow (NPE) training
# --------------------------------------------------------------------------- #

def _train_flow(config, simulator, key, print_every):
    from dmipy_jax.inference.trainer import create_trainer, train_loop

    k_flow, k_train = jax.random.split(key)

    # Adapt simulator to the (key, theta_batch)->signal interface
    def sim_fn(k, theta):
        return simulator.simulate(k, theta)

    def prior_fn(k, n):
        return simulator.prior_sampler(k, n)

    trainer = create_trainer(
        flow_key=k_flow,
        theta_dim=config.theta_dim,
        signal_dim=simulator.signal_dim,
        simulator=sim_fn,
        prior_sampler=prior_fn,
        learning_rate=config.learning_rate,
        hidden_dim=config.hidden_dim,
        num_layers=config.depth,
    )

    trained = train_loop(
        trainer,
        key=k_train,
        num_steps=config.n_steps,
        batch_size=config.batch_size,
        noise_std=config.noise_sigma,
        print_every=print_every,
    )

    # Extract the flow for downstream use
    return trained.flow, []
