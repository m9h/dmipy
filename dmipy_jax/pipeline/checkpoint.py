"""
Checkpoint serialisation for trained SBI models.

Uses ``equinox.tree_serialise_leaves`` for the model and a JSON sidecar
for the pipeline configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import jax
import equinox as eqx

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.inference.mdn import MixtureDensityNetwork


def save_checkpoint(
    model: eqx.Module,
    config: SBIPipelineConfig,
    path: str,
) -> None:
    """Persist a trained model and its config to disk.

    Creates two files next to each other::

        <path>.eqx          – Equinox serialised leaves
        <path>.config.json  – Pipeline config

    Parameters
    ----------
    model : eqx.Module
        Trained MDN or flow module.
    config : SBIPipelineConfig
        The config that was used for training.
    path : str
        Base path (without extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    eqx_path = path.with_suffix(".eqx")
    cfg_path = path.with_name(path.stem + ".config.json")

    eqx.tree_serialise_leaves(str(eqx_path), model)

    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=_json_default)


def load_checkpoint(
    path: str,
    *,
    key: jax.Array | None = None,
) -> Tuple[eqx.Module, SBIPipelineConfig]:
    """Restore a model + config from disk.

    Parameters
    ----------
    path : str
        Base path used during ``save_checkpoint``.
    key : jax.Array, optional
        PRNG key for re-initialising the skeleton model (needed by Equinox
        deserialise).  Defaults to ``PRNGKey(0)``.

    Returns
    -------
    model : eqx.Module
    config : SBIPipelineConfig
    """
    path = Path(path)
    eqx_path = path.with_suffix(".eqx")
    cfg_path = path.with_name(path.stem + ".config.json")

    with open(cfg_path) as f:
        config = SBIPipelineConfig.from_dict(json.load(f))

    if key is None:
        key = jax.random.PRNGKey(0)

    skeleton = _build_skeleton(config, key)
    model = eqx.tree_deserialise_leaves(str(eqx_path), skeleton)
    return model, config


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_skeleton(config: SBIPipelineConfig, key: jax.Array) -> eqx.Module:
    """Create an uninitialised model with the right structure."""
    if config.inference_mode == "mdn":
        signal_dim = len(config.acquisition.get("bvalues", [0] * 32))
        return MixtureDensityNetwork(
            in_features=signal_dim,
            out_features=config.theta_dim,
            num_components=config.n_components,
            width_size=config.hidden_dim,
            depth=config.depth,
            key=key,
        )
    elif config.inference_mode == "flow":
        from dmipy_jax.inference.trainer import create_trainer
        signal_dim = len(config.acquisition.get("bvalues", [0] * 32))
        # Build a dummy trainer to extract the flow skeleton
        trainer = create_trainer(
            flow_key=key,
            theta_dim=config.theta_dim,
            signal_dim=signal_dim,
            simulator=lambda k, t: t,  # dummy
            prior_sampler=lambda k, n: jax.numpy.zeros((n, config.theta_dim)),
            learning_rate=config.learning_rate,
            hidden_dim=config.hidden_dim,
            num_layers=config.depth,
        )
        return trainer.flow
    else:
        raise ValueError(f"Unknown inference_mode: {config.inference_mode!r}")


def _json_default(obj):
    """Fallback serialiser for numpy / jax arrays in config dicts."""
    import numpy as np
    import jax.numpy as jnp

    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
