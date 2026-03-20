from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import flowjax.distributions as dist
import flowjax.flows as flows
import flowjax.bijections as bij
from typing import Callable, Any, Optional, Tuple, Union
from jaxtyping import Array, Float, PRNGKeyArray


def create_trainer(
    flow_key: PRNGKeyArray,
    theta_dim: int,
    signal_dim: int,
    simulator: Callable,
    prior_sampler: Callable,
    learning_rate: Union[float, optax.Schedule] = 1e-4,
    hidden_dim: int = 64,
    num_layers: int = 4,
    flow_type: str = "affine",
    knots: int = 8,
) -> Tuple[Any, optax.GradientTransformation]:
    """Create a conditional normalising flow and optimizer.

    Parameters
    ----------
    learning_rate : float or optax.Schedule
        Either a constant learning rate or an optax schedule object
        (e.g. from ``optax.cosine_decay_schedule``).
    flow_type : str
        ``"affine"`` (default) or ``"spline"`` for rational-quadratic spline.
    knots : int
        Number of knots when ``flow_type="spline"``.
    """
    base_dist = dist.StandardNormal((theta_dim,))

    transformer: Optional[bij.AbstractBijection] = None
    if flow_type == "spline":
        transformer = bij.RationalQuadraticSpline(knots=knots, interval=3.0)

    flow = flows.masked_autoregressive_flow(
        key=flow_key,
        base_dist=base_dist,
        cond_dim=signal_dim,
        transformer=transformer,
        flow_layers=num_layers,
        nn_width=hidden_dim,
        nn_depth=2,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
    )
    return flow, optimizer


def train_loop(
    flow,
    optimizer,
    *,
    simulator: Callable,
    prior_sampler: Callable,
    key: PRNGKeyArray,
    num_steps: int,
    batch_size: int,
    noise_std: float = 0.0,
    print_every: int = 100,
    step_callback: Callable = None,
    trackio_run: bool = False,
):
    """Train a conditional flow via maximum-likelihood on simulated data.

    Parameters
    ----------
    trackio_run : bool
        If *True*, log loss to Trackio (must already be initialised via
        ``trackio.init(...)`` before calling this function).
    """
    if trackio_run:
        import trackio

    opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(flow, opt_state, theta, signal):
        def loss_fn(flow):
            log_probs = flow.log_prob(theta, condition=signal)
            return -jnp.mean(log_probs)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
        # Extract only inexact (float) arrays for the optimizer
        grads_f = eqx.filter(grads, eqx.is_inexact_array)
        params_f = eqx.filter(flow, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads_f, opt_state, params_f)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    curr_key = key
    losses = []
    t0 = time.time()

    for i in range(num_steps):
        if step_callback is not None:
            step_callback(i, num_steps)
        k_step, curr_key = jax.random.split(curr_key)
        k1, k2, k3 = jax.random.split(k_step, 3)

        theta = prior_sampler(k1, batch_size)
        signal = simulator(k2, theta)
        if noise_std > 0:
            signal = signal + jax.random.normal(k3, signal.shape) * noise_std

        flow, opt_state, loss = step(flow, opt_state, theta, signal)
        loss_val = float(loss)
        losses.append(loss_val)

        if trackio_run:
            trackio.log({"step": i, "loss": loss_val})

        if i % print_every == 0:
            print(f"[Flow] step {i}/{num_steps}  loss={loss:.4f}")

    elapsed = time.time() - t0
    print(f"[Flow] Training done. {num_steps} steps in {elapsed:.1f}s "
          f"({num_steps / elapsed:.0f} steps/s)")
    return flow, losses
