# OpenLIFU Heterogeneous Skull Modeling

Documentation for the differentiable transcranial focused ultrasound simulation
layer bridging sbi4dwi and [OpenLIFU](https://github.com/OpenwaterHealth/openlifu-python).

## Documents

- [**proposal.md**](proposal.md) — Formal proposal for Openwater Health describing the architecture, integration path, hackathon plan, and benefits
- [**research-notes.md**](research-notes.md) — Technical research covering all 59 Openwater repos, the TUS simulation landscape (BabelBrain, NDK, PRESTUS, j-Wave, etc.), API surfaces, acoustic property reference values, and WAND/SCI head model context

## Code

### sbi4dwi (R&D layer, JAX-native)

| Module | Purpose |
|--------|---------|
| `dmipy_jax/biophysics/acoustic.py` | Tissue label to acoustic property mapping (ITRUSST values) + HU-based continuous estimation |
| `dmipy_jax/biophysics/mesh_rasterizer.py` | Tetrahedral mesh to regular grid rasterization (SCI head model) |
| `dmipy_jax/biophysics/jwave_adapter.py` | j-Wave differentiable simulation adapter |
| `dmipy_jax/biophysics/tus_optimizer.py` | Gradient-based delay optimization through skull |
| `dmipy_jax/biophysics/tus_solution_export.py` | Export to openlifu Solution format |

### openlifu-python (clinical bridge)

| Module | Purpose |
|--------|---------|
| [`HeterogeneousSkullSegmentation`](https://github.com/m9h/openlifu-python/tree/feature/heterogeneous-skull-segmentation) | SegmentationMethod subclass for heterogeneous tissue modeling |

## Results (Modal A100, 2026-04-04)

Full experiment suite on SCI Institute head model (Warner et al. 2019), 208x256x256 at 1mm, 8 tissue types. **8/9 experiments passed.** Run with: `modal run scripts/modal_experiments.py`

### SCI Head 2D Simulation

| Metric | Value |
|--------|-------|
| Skull attenuation at 400kHz | **93.0%** (matches cortical bone literature) |
| Water p_target | 2.13e-4 Pa |
| Skull p_target | 1.49e-5 Pa |

### Multi-Element Array Optimization (20 iterations)

| Array | Loss Start | Loss End | Improvement | Time/iter |
|-------|-----------|----------|-------------|-----------|
| 4 elements | -0.0904 | -0.0907 | 1.0x | 2.1s |
| 16 elements | -0.1216 | -0.1545 | **1.3x** | 1.8s |
| 32 elements | -0.2991 | -0.4138 | **1.4x** | 1.8s |

More elements = higher baseline pressure + more optimization headroom.

### Multi-Target Attenuation

| Target | Attenuation | p_skull |
|--------|-------------|---------|
| Shallow cortex | 55.4% | 0.0187 |
| Mid-brain | 55.3% | 0.0163 |
| Deep thalamus | 53.9% | 0.0149 |

Attenuation is depth-dependent but consistent (53-55% for this slab geometry).

### Frequency Comparison

| Frequency | Attenuation | p_skull |
|-----------|-------------|---------|
| **180 kHz** | **32.0%** | 0.0157 |
| 400 kHz | 55.1% | 0.0154 |
| 1 MHz | 34.3% | 0.0366 |

Lower frequency (180kHz) has best skull penetration, confirming OpenLIFU's frequency range is well-chosen.

### Grid Convergence

| Spacing | Grid Size | p_max | p_target |
|---------|-----------|-------|----------|
| 0.4mm | 32^2 | 0.502 | 0.0751 |
| 0.2mm | 64^2 | 0.311 | 0.0377 |
| 0.1mm | 128^2 | 0.188 | 0.0188 |

Pressure decreases with refinement (expected — finer grid resolves more diffraction). Values converging.

### Sensitivity Analysis

| Region | Mean |dp/dc| | Ratio to Water |
|--------|--------------|----------------|
| Skull | 1.41e-7 | **232x** |
| Brain | 9.27e-7 | 1523x |
| Water | 6.09e-10 | 1x (baseline) |

Skull region sensitivity is 232x higher than water — confirms that skull property uncertainty dominates focal accuracy.

### Helmholtz Solver

Failed: j-Wave's `helmholtz_solver` requires `FourierSeries` input, not `OnGrid`. Needs input format fix.

## Context

- **NeuroTechX Global NeuroHack** — hackathon project
- **Openwater Health** — collaboration proposal
- **SCI Institute head model** — test dataset (Warner et al. 2019, CC-BY 4.0)
- **Modal A100** — cloud GPU for simulation and optimization
- **DGX Spark** — local GPU compute target
