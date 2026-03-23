# Dmipy Autoresearch: Neural Posterior Estimation for Diffusion MRI Microstructure

## Research Goal

Improve neural posterior estimation (NPE) for diffusion MRI microstructure
parameters using the Ball + 2-Stick forward model. The posterior maps noisy
multi-shell diffusion signals to tissue parameters (diffusivities, volume
fractions, fiber orientations).

**You are NOT starting from scratch.** A spline-based normalizing flow (NPE)
already achieves ~3.2° median fiber orientation error. Your job is to beat
that number — either by improving accuracy, reducing training cost, or both.

## Primary Metric

**Median fiber orientation error (degrees)** for the primary fiber.
Lower is better. This is reported as `fiber_error_deg` in RESULT| lines.

## Secondary Metrics

- `d_stick_r`: Pearson correlation for stick diffusivity (target: >0.95)
- `f1_r`: Pearson correlation for primary volume fraction (target: >0.95)
- `final_loss`: terminal training loss
- `train_time_s`: wall-clock training time

## Current Best: Spline NPE (3.2° baseline)

The flow-based architecture already achieves **~3.2° median Fiber 1 error**.
Do not try to rediscover this from scratch. Start from the flow architecture
in `experiment.py` and make targeted improvements.

### Configuration (commit 7861043)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `inference_mode` | `"flow"` | FlowJAX normalizing flow |
| `flow_type` | `"spline"` | Rational-quadratic spline transformer |
| `knots` | 8 | Spline knots per dimension |
| `num_layers` | 10 | Masked autoregressive flow layers |
| `hidden_dim` | 128 | Hidden units per layer (2-layer MLPs inside each flow layer) |
| `learning_rate` | 3e-4 | Adam optimizer |
| `lr_schedule` | `"cosine"` | Cosine decay |
| `batch_size` | 512 | Per-step training batch |
| `n_steps` | 200,000 | Full training budget (expensive!) |
| `noise_type` | `"rician"` | Applied inside sim_fn, not by train_loop |
| `snr_range` | (10, 50) | Variable SNR augmentation during training |
| `snr` (eval) | 30 | Fixed SNR for evaluation |
| `n_posterior_samples` | 500 | Samples drawn for median point estimate |
| `n_eval` | 500–2000 | Test set size |

### Key design choices that enabled 3.2°

1. **Spline transformer** (`RationalQuadraticSpline(knots=8, interval=3.0)`) —
   much more expressive than affine coupling layers.
2. **b0-normalisation** in the simulator wrapper — signals divided by mean b0
   value before feeding to the flow.
3. **Label-switching symmetry broken** in the prior — canonical ordering
   enforces f1 >= f2 and swaps corresponding orientations.
4. **Hemisphere canonicalization** — orientation vectors constrained to z >= 0.
5. **Variable SNR augmentation** (10–50) during training, fixed SNR 30 at eval.
6. **Clip + unit-normalise + median** posterior aggregation — flow samples
   clipped to prior bounds, orientation vectors renormalised to unit sphere,
   then median taken across 500 samples.
7. **Noise applied inside sim_fn** — `noise_std=0.0` in `train_loop` because
   Rician noise is applied in the simulator wrapper, not as additive Gaussian.

## Baselines

| Method | Fiber 1 Median Error | Training Steps | Notes |
|--------|---------------------|----------------|-------|
| Flow (spline NPE) | **~3.2 deg** | 200k | **Current best** — see config above |
| MLP score-based | ~15.5 deg | 30k | Fast training, plain MLP backbone |
| MDN (10-comp) | ~5–8 deg | 30k | Fast, but limited expressiveness |

## Target

Beat the **3.2° spline NPE baseline**. Concretely:

- **Accuracy target**: < 3.0° median fiber orientation error (ideally < 2.5°)
- **Efficiency target**: achieve comparable accuracy (~3.2°) in < 50k steps
- **Stretch goal**: both — < 3.0° in < 50k steps

Promising directions to explore (not exhaustive):

- Deeper or wider flow architectures (more layers, wider hidden dim)
- Alternative flow transforms (coupling layers, continuous normalizing flows)
- Improved posterior aggregation (geometric median on S², Fréchet mean)
- Learning rate schedules (warmup + cosine, cyclical)
- Curriculum noise schedules (start high SNR, anneal to low)
- Embedding network for the conditioning signal (pre-MLP, attention)
- Mixture of flows or flow ensembles
- Score-based / diffusion posterior (already scaffolded in codebase, currently ~15.5°)
- Hybrid architectures combining flow expressiveness with MDN speed

## Data

Synthetic Ball + 2-Stick signals with HCP-like 90-direction multi-shell
acquisition (b = 0/1000/2000/3000 s/mm²). Rician noise at SNR 10–50
(training), fixed SNR 30 (evaluation).

## Constraints

- Must use `prepare.py` API (do not modify)
- Each experiment has a 10-minute timeout
- Results must call `print_result()` and `log_result()`
- The baseline experiment is in `experiment.py` — modify or extend it
