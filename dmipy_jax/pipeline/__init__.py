"""
Production SBI Pipeline for dmipy-JAX.

Provides a unified train -> serialize -> deploy workflow for
simulation-based inference of diffusion MRI microstructure parameters.
"""

from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint
from dmipy_jax.pipeline.deploy import SBIPredictor
from dmipy_jax.pipeline.ensemble import (
    train_ensemble,
    EnsemblePredictor,
    save_ensemble,
    load_ensemble,
)
from dmipy_jax.pipeline.conformal import (
    ConformalCalibrator,
    ConformalResult,
    calibrate_from_predictions,
    predict_intervals_from_predictions,
)

__all__ = [
    "SBIPipelineConfig",
    "ModelSimulator",
    "train_sbi",
    "save_checkpoint",
    "load_checkpoint",
    "SBIPredictor",
    "train_ensemble",
    "EnsemblePredictor",
    "save_ensemble",
    "load_ensemble",
    "ConformalCalibrator",
    "ConformalResult",
    "calibrate_from_predictions",
    "predict_intervals_from_predictions",
]
