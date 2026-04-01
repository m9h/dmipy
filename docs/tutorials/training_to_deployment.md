# End-to-End Tutorial: Training to Deployment

This tutorial walks through the complete SBI4DWI workflow -- from defining a
biophysical forward model and simulating training data, through training a
neural posterior estimator, to deploying the trained model on real NIfTI
diffusion-weighted images. Along the way we cover validation diagnostics,
uncertainty quantification, and advanced multi-fidelity strategies.

```{contents}
:depth: 3
:local:
```

## Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/) installed
- SBI4DWI installed: `uv sync` from the repository root
- A GPU is recommended but not required (JAX will fall back to CPU)
- Familiarity with diffusion MRI basics (b-values, gradient directions, signal
  attenuation)

```bash
git clone https://github.com/m9h/sbi4dwi.git
cd sbi4dwi
uv sync
```

---

## 1. Introduction: What is Simulation-Based Inference for dMRI?

In conventional diffusion MRI microstructure imaging, a biophysical model
$S(\boldsymbol{\theta}, \mathbf{q})$ maps tissue parameters $\boldsymbol{\theta}$
(e.g. axon density, fibre orientation, diffusivity) and an acquisition scheme
$\mathbf{q}$ (b-values, gradient directions, pulse timings) to a predicted
signal. Parameter estimation traditionally proceeds by fitting this forward
model to each voxel independently -- a nonlinear optimisation problem that is
slow, sensitive to initialisation, and yields only point estimates.

**Simulation-based inference (SBI)** inverts this paradigm
{cite}`cranmer2020frontier`. Instead of fitting each voxel at test time, we:

1. **Simulate** a large training set of $(\boldsymbol{\theta}, \mathbf{x})$
   pairs by sampling parameters from a prior and running them through the
   forward model with realistic noise.
2. **Train** a neural density estimator (MDN, normalising flow, or score
   network) to approximate the posterior
   $p(\boldsymbol{\theta} \mid \mathbf{x})$.
3. **Deploy** the trained network as a fast amortised inference engine:
   given a new observation $\mathbf{x}_\text{obs}$, a single forward pass
   through the network yields the full posterior distribution -- no
   iterative optimisation required.

This approach, termed Neural Posterior Estimation (NPE) by
Papamakarios & Murray (2016) and applied to dMRI by
Manzano-Patron et al. (2025), brings several advantages:

- **Speed**: Whole-brain inference in seconds on GPU (voxel-wise forward
  passes through the network, fully vectorised via `jax.vmap`).
- **Uncertainty**: Full posterior distributions rather than point estimates.
- **Flexibility**: The forward simulator need not be differentiable -- any
  simulator that produces signals given parameters can be used.

SBI4DWI implements this pipeline in JAX with Equinox modules, supporting
three posterior estimation architectures: Mixture Density Networks (MDN),
normalising flows (via FlowJAX), and score-based diffusion models.

---

## 2. Defining the Forward Model

### 2.1 Setting Up the Acquisition Scheme

The acquisition scheme describes the MRI experiment: b-values, gradient
directions, and pulse timings. In SBI4DWI, this is represented by
{class}`~dmipy_jax.acquisition.JaxAcquisition`, a JAX-compatible dataclass
registered as a pytree node.

```python
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.acquisition import JaxAcquisition

# Multi-shell acquisition: 6 b=0 + 30 directions at b=1000 + 60 at b=2000
# b-values in SI units: s/m^2 (FSL uses s/mm^2 -- multiply by 1e6)
b0 = np.zeros(6)
b1000 = np.ones(30) * 1000e6      # 1000 s/mm^2 -> 1e9 s/m^2
b2000 = np.ones(60) * 2000e6      # 2000 s/mm^2 -> 2e9 s/m^2
bvals = np.concatenate([b0, b1000, b2000])

# Random gradient directions on the unit sphere
rng = np.random.default_rng(42)
n_total = len(bvals)
raw = rng.standard_normal((n_total, 3))
raw[: len(b0)] = 0.0  # b=0 volumes have no gradient
norms = np.linalg.norm(raw, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-8)
bvecs = raw / norms

acq = JaxAcquisition(
    bvalues=bvals,
    gradient_directions=bvecs,
    delta=10.3e-3,   # pulse duration (s)
    Delta=43.1e-3,   # pulse separation (s)
)
print(f"Acquisition: {acq.bvalues.shape[0]} measurements, "
      f"shells at b = {np.unique(np.round(bvals / 1e6, -1))} s/mm^2")
```

:::{note}
**Unit convention**: SBI4DWI stores b-values in SI (s/m$^2$) internally.
When loading FSL-format `.bval` files (which use s/mm$^2$), remember to
multiply by $10^6$. The `JaxAcquisition.from_bids_data()` class method
handles this automatically for BIDS-formatted data.
:::

### 2.2 Composing a Signal Model

SBI4DWI provides analytical signal models as Equinox modules. These can be
composed into multi-compartment models with volume fractions and orientation
distributions.

For this tutorial we use a two-compartment "Ball and Stick" model -- one of
the simplest models that captures the essential contrast between free water
(Ball) and oriented axonal diffusion (Stick):

$$
S(\mathbf{q}) = f \cdot S_\text{Stick}(\mathbf{q}; d_\parallel, \hat{\mathbf{n}})
+ (1 - f) \cdot S_\text{Ball}(\mathbf{q}; d_\text{iso})
$$

```python
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.signal_models.gaussian_models import G1Ball
from dmipy_jax.core.modeling_framework import compose_models

# Compose the model
model = compose_models([C1Stick(), G1Ball()])
```

### 2.3 Defining the Forward Function and Prior

The SBI pipeline needs a callable `forward_fn(params_flat, acquisition)`
that maps a flat parameter vector to a signal vector, plus parameter ranges
for the prior:

```python
# Parameter layout for Ball+Stick:
#   [0]: f        - volume fraction of the Stick compartment
#   [1]: mu_theta - polar angle of fibre orientation (rad)
#   [2]: mu_phi   - azimuthal angle of fibre orientation (rad)
#   [3]: d_par    - parallel diffusivity of Stick (m^2/s)
#   [4]: d_iso    - isotropic diffusivity of Ball (m^2/s)

parameter_names = ["f", "mu_theta", "mu_phi", "d_par", "d_iso"]
parameter_ranges = {
    "f":        (0.05, 0.95),
    "mu_theta": (0.0, jnp.pi),
    "mu_phi":   (0.0, 2 * jnp.pi),
    "d_par":    (0.5e-9, 3.0e-9),
    "d_iso":    (1.0e-9, 3.5e-9),
}

def forward_fn(params, acquisition):
    """Map flat parameter vector to predicted signal."""
    f = params[0]
    mu = params[1:3]       # (theta, phi)
    d_par = params[3]
    d_iso = params[4]

    # Convert spherical orientation to Cartesian unit vector
    n = jnp.array([
        jnp.sin(mu[0]) * jnp.cos(mu[1]),
        jnp.sin(mu[0]) * jnp.sin(mu[1]),
        jnp.cos(mu[0]),
    ])

    # Stick signal: exp(-b * d_par * (g . n)^2)
    gn = acquisition.gradient_directions @ n
    stick_signal = jnp.exp(-acquisition.bvalues * d_par * gn**2)

    # Ball signal: exp(-b * d_iso)
    ball_signal = jnp.exp(-acquisition.bvalues * d_iso)

    return f * stick_signal + (1.0 - f) * ball_signal
```

---

## 3. Simulation: Generating Training Data

### 3.1 The ModelSimulator

{class}`~dmipy_jax.pipeline.simulator.ModelSimulator` wraps the forward
function, prior, and noise model into a unified interface that the training
pipeline consumes:

```python
from dmipy_jax.pipeline.simulator import ModelSimulator

simulator = ModelSimulator(
    forward_fn=forward_fn,
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    acquisition=acq,
    noise_type="rician",    # Realistic MRI noise model
    snr=30.0,               # Signal-to-noise ratio at b=0
    snr_range=(10, 50),     # Variable SNR augmentation during training
)

print(f"Signal dimension: {simulator.signal_dim}")
print(f"Parameter dimension: {simulator.theta_dim}")
```

The key method is `sample_and_simulate()`, which:
1. Samples parameter vectors from the prior (uniform within `parameter_ranges`).
2. Runs the forward model on each sample (`jax.vmap`-accelerated).
3. Adds Rician noise (matching clinical MRI noise characteristics).
4. **b0-normalises** the noisy signals (divides by mean b=0 signal).

```python
key = jax.random.key(0)
theta, signals = simulator.sample_and_simulate(key, n=5)

print(f"theta shape: {theta.shape}")    # (5, 5)
print(f"signals shape: {signals.shape}")  # (5, 96)
print(f"Signal range: [{float(signals.min()):.3f}, {float(signals.max()):.3f}]")
```

:::{important}
**b0 normalisation must match between training and deployment.** The
`ModelSimulator.sample_and_simulate()` method normalises signals by their
mean b=0 value. The same normalisation is applied at inference time inside
`SBIPredictor.predict_volume()`. If these do not match, the network will see
a different input distribution and predictions will be unreliable.
:::

### 3.2 Building a SimulationLibrary

For large-scale experiments or when you want to reuse training data across
runs, the {class}`~dmipy_jax.library.storage.SimulationLibrary` provides
HDF5-backed persistence:

```python
from dmipy_jax.library.generator import LibraryGenerator
from dmipy_jax.library.storage import SimulationLibrary

# Generate 200,000 entries in GPU-friendly chunks
gen = LibraryGenerator(simulator, chunk_size=50_000)
params, signals = gen.generate(n_entries=200_000, key=jax.random.key(1))

# Wrap in a SimulationLibrary and save
library = SimulationLibrary(
    params=params,
    signals=signals,
    parameter_names=parameter_names,
    metadata={"model": "BallStick", "snr": 30.0},
)
library.save_hdf5("data/ballstick_library.h5")
print(f"Library saved: {library.n_entries} entries, "
      f"{library.theta_dim}D params, {library.signal_dim}D signals")

# Reload later
library = SimulationLibrary.load_hdf5("data/ballstick_library.h5")
```

The HDF5 schema stores:
- `/params` -- `(N, P)` parameter array
- `/signals` -- `(N, M)` signal array
- `attrs:parameter_names` -- ordered parameter names
- `attrs:*` -- arbitrary metadata

### 3.3 Noise and SNR Augmentation

Realistic training data should span the range of noise levels encountered
clinically. SBI4DWI supports:

- **Fixed SNR**: `snr=30.0` adds noise with $\sigma = 1/\text{SNR}$.
- **Variable SNR**: `snr_range=(10, 50)` samples a random SNR for each
  training example uniformly from the given range.
- **Curriculum noise**: `curriculum_noise=True` starts training with high
  SNR (easy examples) and gradually introduces low-SNR samples. This
  stabilises early training.

```python
# Variable-SNR simulator for robust training
simulator_robust = ModelSimulator(
    forward_fn=forward_fn,
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    acquisition=acq,
    noise_type="rician",
    snr=30.0,
    snr_range=(10, 50),  # Random SNR per sample
)
```

---

## 4. Training

### 4.1 Configuring the Pipeline

All training hyperparameters live in a single
{class}`~dmipy_jax.pipeline.config.SBIPipelineConfig` dataclass:

```python
from dmipy_jax.pipeline.config import SBIPipelineConfig

config = SBIPipelineConfig(
    # Model identity
    model_name="BallStick",
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,

    # Architecture
    inference_mode="mdn",        # "mdn", "flow", or "score"
    architecture="residual",     # "mlp" or "residual" (with skip connections)
    n_components=8,              # Gaussian mixture components (MDN only)
    hidden_dim=256,              # Width of hidden layers
    depth=4,                     # Number of layers / residual blocks
    activation="gelu",           # "relu", "gelu", or "silu"

    # Noise (must match simulator)
    noise_type="rician",
    snr=30.0,
    snr_range=(10, 50),
    curriculum_noise=True,

    # Training
    learning_rate=1e-3,
    lr_schedule="warmup_cosine",  # "constant", "cosine", "warmup_cosine"
    warmup_steps=500,
    batch_size=512,
    n_steps=10_000,

    # Regularisation
    use_ema=True,                # Exponential moving average of weights
    ema_decay=0.999,
    val_fraction=0.1,            # Fraction of each batch for validation
    patience=2000,               # Early stopping (0 = disabled)

    # Reproducibility
    seed=42,
    checkpoint_path="checkpoints/ballstick_mdn",
)
```

### 4.2 Training an MDN Posterior

The {func}`~dmipy_jax.pipeline.train.train_sbi` function is the main
training entry point. It handles architecture construction, optimiser setup,
and the training loop:

```python
from dmipy_jax.pipeline.train import train_sbi

model, losses = train_sbi(
    config,
    simulator,
    print_every=1000,
)
# [MDN] step 0/10000  train=12.3456  val=12.4321  lr=0.00e+00
# [MDN] step 1000/10000  train=2.1543  val=2.2104  lr=9.75e-04
# ...
# [MDN] Training done. 10000 steps in 45.2s (221 steps/s)
```

The returned `model` is a `_NormalisedMDN` (or `_NormalisedFlow`) wrapper
that automatically denormalises outputs back to physical units. The `losses`
list contains per-step training NLL values for convergence monitoring.

### 4.3 Training a Normalising Flow Posterior

For richer, more flexible posteriors, switch to normalising flows. Neural
spline flows in particular handle multi-modal posteriors (e.g. orientation
ambiguity) well:

```python
flow_config = SBIPipelineConfig(
    model_name="BallStick",
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    inference_mode="flow",
    flow_type="spline",          # "affine" or "spline"
    knots=8,                     # Knots for rational-quadratic spline
    hidden_dim=128,
    depth=6,                     # Number of flow layers
    learning_rate=5e-4,
    lr_schedule="warmup_cosine",
    warmup_steps=1000,
    batch_size=512,
    n_steps=50_000,
    noise_type="rician",
    snr=30.0,
    snr_range=(10, 50),
    seed=42,
    checkpoint_path="checkpoints/ballstick_flow",
)

flow_model, flow_losses = train_sbi(flow_config, simulator, print_every=5000)
```

Under the hood, `train_sbi` dispatches to either `_train_mdn()` or
`_train_flow()` based on `config.inference_mode`. The flow backend uses
FlowJAX's `masked_autoregressive_flow` with optional spline transformers,
trained via conditional maximum likelihood.

### 4.4 Monitoring Convergence

Plot the training loss to check for convergence:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(losses, alpha=0.3, label="Raw")
# Smoothed loss (rolling mean)
window = 100
smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
ax.plot(range(window - 1, len(losses)), smoothed, label=f"Smoothed ({window})")
ax.set_xlabel("Training step")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("MDN Training Loss")
ax.legend()
plt.tight_layout()
```

Good convergence indicators:
- Loss decreases monotonically during the first ~1000 steps.
- The validation loss (printed during training) tracks the training loss
  without large divergence (no overfitting).
- The smoothed loss plateaus.

### 4.5 Saving and Loading Checkpoints

SBI4DWI uses Equinox's tree serialisation with a JSON config sidecar:

```python
from dmipy_jax.pipeline.checkpoint import save_checkpoint, load_checkpoint

# Save
save_checkpoint(model, config, "checkpoints/ballstick_mdn")
# Creates: checkpoints/ballstick_mdn.eqx
#          checkpoints/ballstick_mdn.config.json

# Load
model_loaded, config_loaded = load_checkpoint("checkpoints/ballstick_mdn")
```

For sharing models, push to the Hugging Face Hub:

```python
from dmipy_jax.pipeline.checkpoint import push_to_hub, pull_from_hub

# Upload
push_to_hub(
    model, config,
    repo_id="your-username/ballstick-mdn-v1",
    metrics={"final_loss": losses[-1]},
)

# Download and load
model_hub, config_hub = pull_from_hub("your-username/ballstick-mdn-v1")
```

---

## 5. Validation

Before deploying a trained model on real data, it is essential to verify
that the posterior is well-calibrated. SBI4DWI provides two complementary
diagnostics.

### 5.1 Simulation-Based Calibration (SBC)

SBC {cite:p}`talts2018validating` checks whether the posterior has correct
coverage: if we sample $\boldsymbol{\theta}^*$ from the prior, simulate
$\mathbf{x}^*$, and then draw posterior samples
$\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_L \sim
p(\boldsymbol{\theta} \mid \mathbf{x}^*)$, the rank of
$\boldsymbol{\theta}^*$ among the posterior samples should be uniformly
distributed.

```python
from dmipy_jax.pipeline.sbc import SBCDiagnostic

sbc = SBCDiagnostic(
    model=model,
    simulator=simulator,
    n_posterior_samples=200,      # L posterior draws per check
)

sbc_result = sbc.run(
    key=jax.random.key(99),
    n_sbc_samples=1000,           # Number of SBC repetitions
)
#   SBC progress: 100/1000
#   SBC progress: 200/1000
#   ...

sbc_result.print_summary()
# Parameter           Cov@50%  Cov@90%  KS stat    KS p  Pass?
# -----------------------------------------------------------
# f                    0.512    0.903   0.0234   0.6521    yes
# mu_theta             0.498    0.897   0.0312   0.3214    yes
# ...

# Visual diagnostic: rank histograms should look uniform
sbc_result.plot_rank_histograms(save_path="figures/sbc_ranks.png")
```

**Interpreting SBC results:**
- **Coverage@90%** should be close to 0.90 for each parameter.
- **KS p-value > 0.05** indicates the rank distribution is consistent with
  uniformity (the posterior is well-calibrated).
- A U-shaped rank histogram indicates the posterior is **overconfident**
  (too narrow).
- An inverted-U shape indicates the posterior is **underconfident** (too wide).

### 5.2 Posterior Predictive Checks (PPC)

PPC verifies that signals simulated from posterior samples are consistent
with the observed data. This complements SBC by checking data-space fit
quality:

```python
from dmipy_jax.pipeline.ppc import PPCDiagnostic

ppc = PPCDiagnostic(
    model=model,
    simulator=simulator,
    n_posterior_samples=200,
)

ppc_result = ppc.run(key=jax.random.key(123), n_checks=500)
ppc_result.print_summary()
# Posterior Predictive Check Results
# ==================================================
#   Mean RMSE:             0.0231
#   Median RMSE:           0.0198
#   Mean coverage (90%):   91.2%
#   Reduced chi-squared:   1.03  (target: ~1.0)
```

**Interpreting PPC results:**
- **Mean coverage (90%)** should be close to 90% -- this is the fraction
  of signal measurements that fall within the 90% prediction interval.
- **Reduced chi-squared** should be close to 1.0. Values >> 1 indicate
  model misspecification (the forward model does not adequately explain the
  data). Values << 1 indicate the posterior is too wide.

---

## 6. Deployment on Real NIfTI Volumes

### 6.1 Using SBIPredictor

The {class}`~dmipy_jax.pipeline.deploy.SBIPredictor` class handles the full
NIfTI-in / NIfTI-out inference pipeline:

```python
from dmipy_jax.pipeline.deploy import SBIPredictor

predictor = SBIPredictor(model, config)

# Or load from a saved checkpoint:
# predictor = SBIPredictor.from_checkpoint("checkpoints/ballstick_mdn")

results = predictor.predict_volume(
    dwi_path="sub-01/dwi/sub-01_dwi.nii.gz",
    bval_path="sub-01/dwi/sub-01_dwi.bval",
    bvec_path="sub-01/dwi/sub-01_dwi.bvec",
    mask_path="sub-01/dwi/sub-01_brain_mask.nii.gz",
    output_dir="output/sub-01/ballstick_sbi",
    batch_size=4096,            # Voxels per GPU batch
    extract_metrics=True,       # Also compute FA, MD, etc.
)

# results is a dict: {param_name: np.ndarray} with 3D volumes
print(f"Output parameters: {list(results.keys())}")
# ['f', 'mu_theta', 'mu_phi', 'd_par', 'd_iso', 'FA', 'MD']
```

Under the hood, `predict_volume` performs:
1. **Load** the 4D NIfTI DWI volume and brain mask.
2. **Flatten** masked voxels to a 2D array `(N_voxels, N_measurements)`.
3. **b0-normalise** each voxel (dividing by its mean b=0 signal).
4. **Batch inference**: process voxels in chunks through the jitted
   prediction function.
5. **Un-flatten** to 3D parameter maps and save as NIfTI files.

For MDN models, the prediction is the weighted mean of mixture components.
For flow models, 100 posterior samples are drawn and averaged.

### 6.2 Controlling GPU Memory

The `batch_size` parameter controls how many voxels are processed in each
GPU call. For a whole-brain mask (~200,000 voxels), typical values:

| GPU VRAM | Recommended `batch_size` |
|----------|:------------------------:|
| 8 GB     | 2048                     |
| 16 GB    | 4096                     |
| 24 GB    | 8192                     |
| 40+ GB   | 16384                    |

### 6.3 Derived Metrics

When `extract_metrics=True`, the `MultiMetricExtractor` computes standard
diffusion metrics from the fitted parameters:

- **FA** (Fractional Anisotropy), **MD** (Mean Diffusivity) from tensor
  parameters
- **NDI** (Neurite Density Index), **ODI** (Orientation Dispersion Index)
  for NODDI-type models
- **AD** (Axial Diffusivity), **RD** (Radial Diffusivity)

### 6.4 Loading Acquisitions from BIDS

For BIDS-formatted datasets, use the class method:

```python
acq = JaxAcquisition.from_bids_data({
    "bval_file": "sub-01/dwi/sub-01_dwi.bval",
    "bvec_file": "sub-01/dwi/sub-01_dwi.bvec",
    "EchoTime": 0.089,
    "SmallDelta": 10.3e-3,
    "BigDelta": 43.1e-3,
})
```

---

## 7. Uncertainty Quantification

A key advantage of SBI over point-estimation methods is access to the full
posterior distribution. SBI4DWI provides several tools for characterising and
calibrating uncertainty.

### 7.1 Posterior Sampling

For any trained model (MDN or flow), you can draw posterior samples
conditioned on an observed signal:

```python
from dmipy_jax.inference.mdn import sample_posterior

# For a single voxel signal
signal_obs = signals[0]  # (96,)

# MDN: sample from the Gaussian mixture
key = jax.random.key(0)
samples = sample_posterior(model, signal_obs, key, n_samples=1000)
# samples shape: (1000, 5) -- 1000 draws in 5D parameter space

# Flow: use the .sample() method
# samples = flow_model.sample(key, (1000,), condition=signal_obs)
```

Visualise the posterior with corner plots:

```python
import corner

fig = corner.corner(
    np.asarray(samples),
    labels=parameter_names,
    truths=np.asarray(theta[0]),
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
)
fig.savefig("figures/posterior_corner.png", dpi=150)
```

### 7.2 Conformal Prediction Intervals

Conformal prediction provides **distribution-free, finite-sample valid**
coverage guarantees regardless of whether the neural posterior is
well-calibrated. SBI4DWI implements split conformal inference with two
methods:

- **Absolute**: Symmetric intervals around the posterior mean.
- **CQR** (Conformalized Quantile Regression): Adaptive intervals that are
  wider where the model is more uncertain.

```python
from dmipy_jax.pipeline.conformal import ConformalCalibrator

# 1. Generate calibration data
key_cal = jax.random.key(77)
theta_cal, signals_cal = simulator.sample_and_simulate(key_cal, n=5000)

# 2. Calibrate
calibrator = ConformalCalibrator(
    model=model,
    parameter_names=parameter_names,
    n_posterior_samples=500,
)
conf_result = calibrator.calibrate(
    theta_cal=np.asarray(theta_cal),
    signals_cal=np.asarray(signals_cal),
    alpha=0.1,                   # 90% coverage target
    method="cqr",                # or "absolute"
)
conf_result.print_summary()
# Conformal Calibration Summary
#   Method:              cqr
#   Alpha:               0.1
#   Target coverage:     90.0%
#   Empirical coverage:  90.3%
#   Calibration samples: 5000

# 3. Predict intervals for new data
intervals = calibrator.predict_intervals(signals_cal[:10])
# intervals["mean"]  -- point estimates (N, D)
# intervals["lower"] -- lower bounds   (N, D)
# intervals["upper"] -- upper bounds   (N, D)
# intervals["width"] -- interval width (N, D)
```

### 7.3 Out-of-Distribution Detection

The {class}`~dmipy_jax.pipeline.ood.OODDetector` flags voxels where the
input signal falls outside the training distribution. It uses three
complementary scores:

1. **Reconstruction error**: RMSE between the observed signal and the signal
   re-simulated from the posterior mean.
2. **Predictive entropy**: Entropy of the MDN mixing weights (high entropy
   means the model is uncertain about which mixture component to use).
3. **Mahalanobis distance**: Statistical distance of the input signal from
   the reference signal distribution.

```python
from dmipy_jax.pipeline.ood import OODDetector

ood = OODDetector(model=model, simulator=simulator)

# Fit reference distribution from in-distribution samples
ood.fit(key=jax.random.key(55), n_reference=5000)

# Flag OOD voxels in a volume
ood_result = ood.flag_volume(
    signals=jnp.array(signals_cal),
    threshold_percentile=95.0,
)
ood_result.print_summary()
# Out-of-Distribution Detection Results
# ==================================================
#   Total voxels:          5000
#   OOD voxels:            243 (4.9%)
#   Reconstruction RMSE:   mean=0.0312  median=0.0245
#   ...
```

In practice, OOD-flagged voxels should be excluded or marked as unreliable
in downstream analyses (e.g. tractography, group statistics).

### 7.4 Ensemble Inference

Deep ensembles -- training multiple models with different random seeds --
provide improved calibration and uncertainty estimates through model
disagreement:

```python
from dmipy_jax.pipeline.ensemble import train_ensemble, EnsemblePredictor

# Train 5 ensemble members
members = train_ensemble(
    config,
    simulator,
    n_members=5,
    print_every=2000,
)

# Combine for inference
models = [m for m, _ in members]
ensemble = EnsemblePredictor(models, config)
```

The ensemble prediction averages point estimates across members and
quantifies uncertainty via inter-member disagreement.

---

## 8. Advanced: Multi-Fidelity SBI

For models where the analytical forward model is an approximation of the
true physics (e.g. simplified cylinder models vs. full Monte Carlo
diffusion simulation), SBI4DWI supports multi-fidelity training data
generation.

### 8.1 Oracle Simulators

Non-differentiable external simulators are wrapped behind the oracle
protocol:

```python
from dmipy_jax.simulation.oracles import get_oracle

# DIPY multi-tensor oracle (fast, Python-native)
dipy_oracle = get_oracle("dipy")

# ReMiDi oracle (high-fidelity mesh-based MC simulation)
# Requires Docker: docker build -f docker/Dockerfile.remidi -t remidi .
# remidi_oracle = get_oracle("remidi")

# MCMRSimulator.jl oracle (Julia-based MC simulation)
# Requires Docker: docker build -f docker/Dockerfile.mcmr -t mcmr .
# mcmr_oracle = get_oracle("mcmr")
```

### 8.2 Generating Oracle Libraries

Oracle simulators produce `SimulationLibrary` datasets:

```python
# Generate a library from the DIPY oracle
if dipy_oracle.check_available():
    oracle_library = dipy_oracle.generate_library(
        n=10_000,
        acquisition=acq,
        parameter_ranges=parameter_ranges,
    )
    oracle_library.save_hdf5("data/dipy_oracle_library.h5")
```

### 8.3 Hybrid Training

The `HybridLibraryGenerator` (referenced in the README) mixes analytical
and oracle-generated data to balance speed and accuracy:

```python
# Conceptual example -- the exact API depends on the oracle adapter
from dmipy_jax.pipeline.oracle_adapter import OracleModelSimulator

# Wrap the oracle library as a ModelSimulator-compatible object
oracle_sim = OracleModelSimulator(
    library=oracle_library,
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    acquisition=acq,
)

# The oracle simulator uses k-NN inverse-distance interpolation
# to generate signals for arbitrary parameter queries
theta_test, signals_test = oracle_sim.sample_and_simulate(
    jax.random.key(0), n=100
)
```

The multi-fidelity strategy is particularly valuable when the analytical
model is fast but approximate, and the oracle is slow but accurate. By
training on a mixture (e.g. 70% analytical, 30% oracle), the neural
posterior learns to correct for the systematic biases of the analytical
model while keeping the bulk of training data cheap to generate.

---

## 9. Putting It All Together

Here is the complete end-to-end pipeline in a single script:

```python
"""Complete SBI4DWI pipeline: simulate -> train -> validate -> deploy."""

import jax
import jax.numpy as jnp
import numpy as np

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.pipeline.simulator import ModelSimulator
from dmipy_jax.pipeline.config import SBIPipelineConfig
from dmipy_jax.pipeline.train import train_sbi
from dmipy_jax.pipeline.checkpoint import save_checkpoint
from dmipy_jax.pipeline.deploy import SBIPredictor
from dmipy_jax.pipeline.sbc import SBCDiagnostic
from dmipy_jax.pipeline.conformal import ConformalCalibrator

# --- 1. Acquisition ---
bvals = np.concatenate([np.zeros(6), np.ones(30) * 1e9, np.ones(60) * 2e9])
bvecs = np.random.default_rng(0).standard_normal((96, 3))
bvecs[:6] = 0
bvecs /= np.maximum(np.linalg.norm(bvecs, axis=1, keepdims=True), 1e-8)
acq = JaxAcquisition(bvalues=bvals, gradient_directions=bvecs,
                     delta=10.3e-3, Delta=43.1e-3)

# --- 2. Forward model ---
parameter_names = ["f", "mu_theta", "mu_phi", "d_par", "d_iso"]
parameter_ranges = {
    "f": (0.05, 0.95), "mu_theta": (0.0, float(jnp.pi)),
    "mu_phi": (0.0, float(2 * jnp.pi)),
    "d_par": (0.5e-9, 3.0e-9), "d_iso": (1.0e-9, 3.5e-9),
}

def forward_fn(params, acquisition):
    f, mu_theta, mu_phi, d_par, d_iso = params
    n = jnp.array([jnp.sin(mu_theta) * jnp.cos(mu_phi),
                    jnp.sin(mu_theta) * jnp.sin(mu_phi),
                    jnp.cos(mu_theta)])
    gn = acquisition.gradient_directions @ n
    return f * jnp.exp(-acquisition.bvalues * d_par * gn**2) + \
           (1 - f) * jnp.exp(-acquisition.bvalues * d_iso)

simulator = ModelSimulator(forward_fn, parameter_names, parameter_ranges,
                           acq, noise_type="rician", snr=30.0,
                           snr_range=(10, 50))

# --- 3. Train ---
config = SBIPipelineConfig(
    model_name="BallStick", parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    inference_mode="mdn", architecture="residual",
    n_components=8, hidden_dim=256, depth=4,
    noise_type="rician", snr=30.0, snr_range=(10, 50),
    curriculum_noise=True, learning_rate=1e-3,
    lr_schedule="warmup_cosine", warmup_steps=500,
    batch_size=512, n_steps=10_000, use_ema=True, seed=42,
)
model, losses = train_sbi(config, simulator, print_every=2000)
save_checkpoint(model, config, "checkpoints/ballstick_mdn")

# --- 4. Validate ---
sbc = SBCDiagnostic(model, simulator, n_posterior_samples=200)
sbc_result = sbc.run(jax.random.key(99), n_sbc_samples=500)
sbc_result.print_summary()

# --- 5. Deploy ---
predictor = SBIPredictor(model, config)
results = predictor.predict_volume(
    "sub-01/dwi/sub-01_dwi.nii.gz",
    "sub-01/dwi/sub-01_dwi.bval",
    "sub-01/dwi/sub-01_dwi.bvec",
    mask_path="sub-01/dwi/sub-01_brain_mask.nii.gz",
    output_dir="output/sub-01",
    extract_metrics=True,
)
print(f"Done. Output maps: {list(results.keys())}")
```

---

## 10. Tips and Troubleshooting

### Architecture Selection

| Scenario | Recommended `inference_mode` |
|:---------|:-----------------------------|
| Quick prototyping, few parameters | `"mdn"` with `architecture="mlp"` |
| Production, complex posteriors | `"flow"` with `flow_type="spline"` |
| Orientation estimation (SO(3)) | `"score"` with equivariant networks |
| Maximum robustness | Ensemble of any of the above |

### Common Pitfalls

1. **Unit mismatch**: b-values must be in SI (s/m$^2$). FSL `.bval` files
   use s/mm$^2$ -- multiply by $10^6$.
2. **b0 normalisation mismatch**: The `ModelSimulator` and `SBIPredictor`
   must apply the same normalisation. This is handled automatically if you
   use the standard pipeline, but be careful with custom forward functions.
3. **Posterior collapse**: If the MDN loss plateaus at a high value, try
   increasing `n_components`, switching to `architecture="residual"`, or
   using a flow.
4. **GPU OOM**: Reduce `batch_size` in training or `batch_size` in
   `predict_volume`.
5. **Noise matching**: Training and test noise distributions must match.
   Use `snr_range` to cover the range of SNR in your clinical data.

### References

- **SBI/NPE framework**: Papamakarios, G. & Murray, I. (2016). Fast
  $\epsilon$-free inference of simulation models with Bayesian conditional
  density estimation. NeurIPS.
- **SBI for dMRI**: Manzano-Patron, J. P. et al. (2025). Uncertainty mapping
  and probabilistic tractography using Simulation-Based Inference in diffusion
  MRI. Medical Image Analysis 103: 103580.
- **dmipy signal models**: Fick, R., Wassermann, D. & Deriche, R. (2019).
  The Dmipy Toolbox. Frontiers in Neuroinformatics 13: 64.
- **SBC validation**: Talts, S. et al. (2018). Validating Bayesian inference
  algorithms with simulation-based calibration. arXiv:1804.06788.
- **Conformal prediction**: Angelopoulos, A. N. & Bates, S. (2023).
  Conformal Prediction: A Gentle Introduction. Foundations and Trends in ML.
- **NODDI**: Zhang, H. et al. (2012). NODDI: practical in vivo neurite
  orientation dispersion and density imaging. NeuroImage 61(4): 1000--1016.
