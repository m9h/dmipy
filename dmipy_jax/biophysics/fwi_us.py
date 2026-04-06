"""
Full Waveform Inversion for Ultrasound (FWI-US).

Uses j-Wave's differentiability to invert for tissue acoustic properties
(sound speed, density) from observed pressure data. This is the acoustic
analog of Electrical Impedance Tomography (EIT) — reconstruct the medium
from measurements of wave propagation through it.

The inversion uses jax.grad through the j-Wave forward model with
optax Adam optimization, following the same pattern as tus_optimizer.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple, Union


def fwi_loss(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    sound_speed: Optional[jnp.ndarray] = None,
    tv_weight: float = 0.01,
) -> float:
    """
    FWI loss: L2 data mismatch + Total Variation regularization.

    Args:
        predicted: Predicted pressure field from forward model.
        observed: Observed (target) pressure field.
        sound_speed: Sound speed field (for TV regularization). If None, no TV.
        tv_weight: Weight for TV regularization term.

    Returns:
        Scalar loss value.
    """
    # Data mismatch (normalized L2)
    data_loss = jnp.mean((predicted - observed) ** 2)

    # Total Variation on sound speed (promotes piecewise-smooth reconstructions)
    if sound_speed is not None and tv_weight > 0:
        tv = _total_variation(sound_speed)
        return data_loss + tv_weight * tv
    return data_loss


def _total_variation(field: jnp.ndarray) -> float:
    """Isotropic Total Variation of a field."""
    tv = 0.0
    for axis in range(field.ndim):
        diff = jnp.diff(field, axis=axis)
        tv = tv + jnp.mean(jnp.abs(diff))
    return tv


def invert_tissue_properties(
    observed_pressure: jnp.ndarray,
    initial_sound_speed: Union[float, jnp.ndarray],
    grid_shape: Tuple[int, ...],
    grid_spacing: float,
    source_position: Tuple[int, ...],
    freq: float,
    density: float = 1000.0,
    n_iters: int = 50,
    lr: float = 1.0,
    tv_weight: float = 0.01,
    t_end: float = 5e-5,
    pml_size: float = 8.0,
) -> Tuple[jnp.ndarray, list]:
    """
    Invert for sound speed field from observed pressure data.

    Uses jax.value_and_grad through j-Wave forward model + optax Adam.

    Args:
        observed_pressure: Target p_max field (from measurement or synthetic).
        initial_sound_speed: Starting guess (scalar or array).
        grid_shape: Simulation grid shape.
        grid_spacing: Grid spacing in metres.
        source_position: Source location (grid indices).
        freq: Source frequency (Hz).
        density: Density (scalar, assumed known).
        n_iters: Optimization iterations.
        lr: Learning rate.
        tv_weight: Total Variation regularization weight.
        t_end: Simulation end time.
        pml_size: PML thickness.

    Returns:
        (reconstructed_sound_speed, loss_history) tuple.
    """
    import optax
    import importlib.util, sys, os
    biophys = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("jwa", os.path.join(biophys, "jwave_adapter.py"))
    jwa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jwa)
    from jwave.geometry import TimeAxis

    # Initialize sound speed field
    if isinstance(initial_sound_speed, (int, float)):
        c = jnp.ones(grid_shape) * initial_sound_speed
    else:
        c = jnp.array(initial_sound_speed)

    # Pre-compute time axis and sources from a reference medium
    domain = jwa.create_domain(grid_shape, grid_spacing)
    med_ref = jwa.create_medium(domain, 1500.0, density, pml_size=pml_size)
    time_axis = TimeAxis.from_medium(med_ref, cfl=0.3, t_end=t_end)
    dt = float(time_axis.dt)

    pos = jnp.array([[source_position[i] * grid_spacing for i in range(len(grid_shape))]])
    sources = jwa.create_sources(domain, pos, freq, jnp.zeros(1), jnp.ones(1), dt=dt, n_cycles=5)

    def loss_fn(c_field):
        # Clip sound speed to physical range for stability
        c_clipped = jnp.clip(c_field, 1000.0, 5000.0)
        medium = jwa.create_medium(domain, c_clipped, density, pml_size=pml_size)
        p_pred = jwa.run_simulation_jax(
            medium, source_position, freq, t_end,
            time_axis=time_axis, sources=sources,
        )
        return fwi_loss(p_pred, observed_pressure, c_clipped, tv_weight)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(c)
    loss_history = []

    for i in range(n_iters):
        loss_val, grads = jax.value_and_grad(loss_fn)(c)
        updates, opt_state = optimizer.update(grads, opt_state)
        c = optax.apply_updates(c, updates)
        loss_history.append(float(loss_val))

    return c, loss_history
