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
from dmipy_jax.pipeline.train import _NormalisedMDN, _NormalisedFlow, _get_activation


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
    """Create an uninitialised model with the right structure.

    The skeleton must match the structure returned by ``train_sbi``, which
    wraps the raw network in a normalisation layer (``_NormalisedMDN`` or
    ``_NormalisedFlow``).
    """
    import jax.numpy as jnp

    signal_dim = len(config.acquisition.get("bvalues", [0] * 32))

    # Compute dummy normalisation bounds (actual values are in the checkpoint)
    lows = jnp.zeros(config.theta_dim)
    spans = jnp.ones(config.theta_dim)

    if config.inference_mode == "mdn":
        if getattr(config, "architecture", "mlp") == "residual":
            from dmipy_jax.inference.mdn import ResidualMDN
            inner = ResidualMDN(
                in_features=signal_dim,
                out_features=config.theta_dim,
                num_components=config.n_components,
                width_size=config.hidden_dim,
                depth=config.depth,
                key=key,
                activation=_get_activation(config.activation),
            )
        else:
            inner = MixtureDensityNetwork(
                in_features=signal_dim,
                out_features=config.theta_dim,
                num_components=config.n_components,
                width_size=config.hidden_dim,
                depth=config.depth,
                key=key,
                activation=_get_activation(config.activation),
            )
        return _NormalisedMDN(inner, lows, spans)
    elif config.inference_mode == "flow":
        from dmipy_jax.inference.trainer import create_trainer
        flow, _ = create_trainer(
            flow_key=key,
            theta_dim=config.theta_dim,
            signal_dim=signal_dim,
            simulator=lambda k, t: t,
            prior_sampler=lambda k, n: jnp.zeros((n, config.theta_dim)),
            learning_rate=config.learning_rate,
            hidden_dim=config.hidden_dim,
            num_layers=config.depth,
            flow_type=config.flow_type,
            knots=config.knots,
        )
        return _NormalisedFlow(flow, lows, spans)
    else:
        raise ValueError(f"Unknown inference_mode: {config.inference_mode!r}")


def push_to_hub(
    model: eqx.Module,
    config: SBIPipelineConfig,
    repo_id: str,
    *,
    commit_message: str = "Upload SBI model checkpoint",
    metrics: dict | None = None,
    private: bool = False,
) -> str:
    """Save a checkpoint and push it to the Hugging Face Hub.

    Parameters
    ----------
    model : eqx.Module
        Trained model.
    config : SBIPipelineConfig
        Pipeline config used for training.
    repo_id : str
        Hub repo id, e.g. ``"mhough/ball2stick-npe-spline-v1"``.
    commit_message : str
        Git commit message for the Hub upload.
    metrics : dict, optional
        Evaluation metrics to store in ``metrics.json``.
    private : bool
        Whether to create a private repo.

    Returns
    -------
    str
        URL of the uploaded repo.
    """
    import tempfile
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "model"
        save_checkpoint(model, config, str(base))

        # Write optional metrics sidecar
        if metrics is not None:
            metrics_path = Path(tmpdir) / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        info = api.upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            commit_message=commit_message,
        )
    print(f"Pushed to https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def pull_from_hub(
    repo_id: str,
    *,
    key: jax.Array | None = None,
) -> Tuple[eqx.Module, SBIPipelineConfig]:
    """Download a checkpoint from the Hugging Face Hub and load it.

    Parameters
    ----------
    repo_id : str
        Hub repo id, e.g. ``"mhough/ball2stick-npe-spline-v1"``.
    key : jax.Array, optional
        PRNG key for skeleton initialisation.

    Returns
    -------
    model : eqx.Module
    config : SBIPipelineConfig
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id)
    return load_checkpoint(str(Path(local_dir) / "model"), key=key)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

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
