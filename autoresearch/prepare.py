"""Prepare module for agentsciml integration with dmipy.

Provides standardized API for experiment scripts to use dmipy_jax
signal models, simulators, and evaluation metrics.
"""
import sys
import os
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing dmipy_jax components (graceful fallback for testing)
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dmipy_jax.acquisition import JaxAcquisition
    from dmipy_jax.pipeline.simulator import ModelSimulator
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    np = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Standardized result from a dmipy autoresearch experiment."""
    commit: str
    model_name: str
    architecture: str
    dataset: str
    fiber1_median_deg: float  # PRIMARY METRIC (lower = better)
    fiber1_mean_deg: float = 0.0
    d_stick_r: float = 0.0  # Pearson correlation for stick diffusivity
    f1_r: float = 0.0  # Pearson correlation for volume fraction
    final_loss: float = 0.0
    train_time_s: float = 0.0
    n_steps: int = 0
    hidden_dim: int = 0
    learning_rate: float = 0.0
    seed: int = 0
    status: str = "ok"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def print_result(result: ExperimentResult) -> None:
    """Print result in RESULT| format for agentsciml sandbox parsing."""
    parts = [
        "RESULT",
        result.model_name,
        result.dataset,
        result.architecture,
        f"fiber_error_deg={result.fiber1_median_deg:.4f}",
        f"d_stick_r={result.d_stick_r:.4f}",
        f"f1_r={result.f1_r:.4f}",
        f"loss={result.final_loss:.6f}",
        f"time={result.train_time_s:.1f}",
        f"steps={result.n_steps}",
    ]
    print("|".join(parts))


def log_result(result: ExperimentResult) -> None:
    """Append result to results.tsv."""
    results_path = Path(__file__).parent / "results.tsv"
    d = asdict(result)
    d.pop("metadata", None)

    write_header = not results_path.exists()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=d.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(d)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angular_error_deg(v1, v2):
    """Compute angular error in degrees between two unit vectors.

    Handles antipodal symmetry (fiber directions are unsigned).
    Works with numpy arrays.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX/numpy not available")
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    cos_angle = np.clip(np.abs(np.dot(v1, v2)), 0, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_fiber_errors(theta_true, preds, n_eval):
    """Compute fiber orientation errors with label-switching resolution.

    Parameters
    ----------
    theta_true : array (n_eval, >=10)
        Ground truth parameters. Columns 4:7 = mu1, 7:10 = mu2.
    preds : array (n_eval, >=10)
        Predicted parameters.
    n_eval : int
        Number of evaluation samples.

    Returns
    -------
    errors1 : array of float
        Per-sample angular error (degrees) for fiber 1.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX/numpy not available")
    theta_np = np.asarray(theta_true)
    preds_np = np.asarray(preds)
    errors1 = []
    for j in range(n_eval):
        m1t = theta_np[j, 4:7]
        m1p = preds_np[j, 4:7]
        m1p = m1p / max(np.linalg.norm(m1p), 1e-8)
        m2t = theta_np[j, 7:10]
        m2p = preds_np[j, 7:10]
        m2p = m2p / max(np.linalg.norm(m2p), 1e-8)
        e11 = angular_error_deg(m1t, m1p)
        e12 = angular_error_deg(m1t, m2p)
        errors1.append(min(e11, e12))
    return np.array(errors1)


def safe_pearson_r(a, b):
    """Pearson correlation, returning 0.0 on NaN."""
    if not HAS_JAX:
        raise RuntimeError("JAX/numpy not available")
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Acquisition and simulator builders
# ---------------------------------------------------------------------------

def make_hcp_acquisition(n_b0=6, n_b1000=30, n_b2000=30, n_b3000=24,
                         seed=0):
    """Create an HCP-like multi-shell acquisition scheme.

    Parameters
    ----------
    n_b0, n_b1000, n_b2000, n_b3000 : int
        Number of directions per shell.
    seed : int
        Random seed for gradient direction generation.

    Returns
    -------
    JaxAcquisition
        Multi-shell acquisition with b = 0, 1e9, 2e9, 3e9 s/m^2.
    """
    if not HAS_JAX:
        raise RuntimeError("JAX not available")

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    def rand_vecs(k, n):
        z = jax.random.normal(k, (n, 3))
        return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    v0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n_b0, 1))
    v1 = rand_vecs(keys[0], n_b1000)
    v2 = rand_vecs(keys[1], n_b2000)
    v3 = rand_vecs(keys[2], n_b3000)

    bvals = jnp.concatenate([
        jnp.zeros(n_b0),
        jnp.full(n_b1000, 1e9),
        jnp.full(n_b2000, 2e9),
        jnp.full(n_b3000, 3e9),
    ])
    bvecs = jnp.concatenate([v0, v1, v2, v3], axis=0)
    return JaxAcquisition(bvalues=bvals, gradient_directions=bvecs)


def build_simulator(acquisition=None, snr=30.0, snr_range=None):
    """Build a Ball + 2-Stick ModelSimulator for synthetic data generation.

    Parameters
    ----------
    acquisition : JaxAcquisition or None
        Acquisition scheme. If None, uses default HCP-like scheme.
    snr : float
        Signal-to-noise ratio for fixed-SNR evaluation.
    snr_range : tuple of (float, float) or None
        If set, training uses variable SNR in this range.

    Returns
    -------
    ModelSimulator
        Ready-to-use simulator with 10-D parameter space:
        [d_ball, d_stick, f1, f2, mu1x, mu1y, mu1z, mu2x, mu2y, mu2z]
    """
    if not HAS_JAX:
        raise RuntimeError("JAX not available")

    if acquisition is None:
        acquisition = make_hcp_acquisition()

    acq = acquisition

    def forward_fn(params, acq):
        d_ball = params[0]
        d_stick = params[1]
        f1 = params[2]
        f2 = params[3]
        f_ball = jnp.clip(1.0 - f1 - f2, 0.0, 1.0)

        mu1 = params[4:7]
        mu1 = mu1 / jnp.maximum(jnp.linalg.norm(mu1), 1e-8)
        mu2 = params[7:10]
        mu2 = mu2 / jnp.maximum(jnp.linalg.norm(mu2), 1e-8)

        cos1 = acq.gradient_directions @ mu1
        cos2 = acq.gradient_directions @ mu2

        s1 = jnp.exp(-acq.bvalues * d_stick * cos1 ** 2)
        s2 = jnp.exp(-acq.bvalues * d_stick * cos2 ** 2)
        s_ball = jnp.exp(-acq.bvalues * d_ball)

        return f1 * s1 + f2 * s2 + f_ball * s_ball

    def _sample_hemisphere(key, n):
        """Sample unit vectors on the z>=0 hemisphere."""
        z = jax.random.normal(key, (n, 3))
        z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
        z = z * jnp.sign(z[:, 2:3] + 1e-8)
        return z

    def prior_fn(key, n):
        keys = jax.random.split(key, 6)
        d_ball = jax.random.uniform(keys[0], (n,), minval=1.0e-9, maxval=3.5e-9)
        d_stick = jax.random.uniform(keys[1], (n,), minval=0.5e-9, maxval=2.5e-9)
        f1_raw = jax.random.uniform(keys[2], (n,), minval=0.1, maxval=0.8)
        f2_raw = jax.random.uniform(keys[3], (n,), minval=0.05, maxval=0.5)
        total = f1_raw + f2_raw
        scale = jnp.minimum(1.0, 0.95 / total)
        f1 = f1_raw * scale
        f2 = f2_raw * scale
        mu1 = _sample_hemisphere(keys[4], n)
        mu2 = _sample_hemisphere(keys[5], n)

        # Canonical ordering: enforce f1 >= f2 to break label-switching
        swap = f1 < f2
        f1_out = jnp.where(swap, f2, f1)
        f2_out = jnp.where(swap, f1, f2)
        swap3 = swap[:, None]
        mu1_out = jnp.where(swap3, mu2, mu1)
        mu2_out = jnp.where(swap3, mu1, mu2)

        return jnp.concatenate([
            d_ball[:, None], d_stick[:, None],
            f1_out[:, None], f2_out[:, None],
            mu1_out, mu2_out,
        ], axis=-1)

    sim = ModelSimulator(
        forward_fn=forward_fn,
        parameter_names=["d_ball", "d_stick", "f1", "f2",
                         "mu1x", "mu1y", "mu1z", "mu2x", "mu2y", "mu2z"],
        parameter_ranges={
            "d_ball": (1.0e-9, 3.5e-9),
            "d_stick": (0.5e-9, 2.5e-9),
            "f1": (0.1, 0.8), "f2": (0.05, 0.5),
            "mu1x": (-1.0, 1.0), "mu1y": (-1.0, 1.0), "mu1z": (0.0, 1.0),
            "mu2x": (-1.0, 1.0), "mu2y": (-1.0, 1.0), "mu2z": (0.0, 1.0),
        },
        acquisition=acq,
        noise_type="rician",
        snr=snr,
        prior_sampler_fn=prior_fn,
    )

    if snr_range is not None:
        sim.snr_range = snr_range

    return sim
