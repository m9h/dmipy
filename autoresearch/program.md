# Dmipy Autoresearch: Neural Posterior Estimation for Diffusion MRI Microstructure

## Research Goal

Optimize neural posterior estimation (NPE) for diffusion MRI microstructure
parameters using the Ball + 2-Stick forward model. The posterior maps noisy
multi-shell diffusion signals to tissue parameters (diffusivities, volume
fractions, fiber orientations).

## Primary Metric

**Median fiber orientation error (degrees)** for the primary fiber.
Lower is better. This is reported as `fiber_error_deg` in RESULT| lines.

## Secondary Metrics

- `d_stick_r`: Pearson correlation for stick diffusivity (target: >0.95)
- `f1_r`: Pearson correlation for primary volume fraction (target: >0.95)
- `final_loss`: terminal training loss
- `train_time_s`: wall-clock training time

## Search Space

- **Network architecture**: MLP depth (4-10), hidden dim (128-512), conditioning dim (64-256)
- **Training hyperparameters**: learning rate (1e-4 to 1e-3), batch size (256-1024), training steps (5k-200k)
- **Noise schedules**: VP-SDE beta_min (0.01-0.5), beta_max (5.0-40.0)
- **Sampler strategies**: SDE step count (50-200), posterior sample count (100-500)
- **Inference method**: MLP score-based, E3-equivariant score-based, MDN, normalizing flow

## Baselines

| Method | Fiber 1 Median Error | Training Steps | Notes |
|--------|---------------------|----------------|-------|
| MLP score | ~15.5 deg | 30k | Fast training, plain MLP backbone |
| Flow (spline) | ~3.2 deg | 200k | Current best, expensive training |
| MDN (10-comp) | ~5-8 deg | 30k | Fast, but limited expressiveness |

## Target

Achieve <5 deg median fiber orientation error with efficient training
(<50k steps), ideally matching or beating the flow baseline quality.

## Data

Synthetic Ball + 2-Stick signals with HCP-like 90-direction multi-shell
acquisition (b=0/1000/2000/3000 s/mm^2). Rician noise at SNR 10-50
(training), fixed SNR 30 (evaluation).

## Constraints

- Must use `prepare.py` API (do not modify)
- Each experiment has a 10-minute timeout
- Results must call `print_result()` and `log_result()`
