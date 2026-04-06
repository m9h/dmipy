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

Full experiment suite on SCI Institute head model (Warner et al. 2019), 208x256x256 at 1mm, 8 tissue types. **9/9 experiments passed.** Run with: `modal run scripts/modal_experiments.py`

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

### 3D Volumetric Simulation (A100, 2026-04-05)

Full 3D simulation through the SCI head model anatomy. Run with: `modal run scripts/modal_3d_validation.py`

| Resolution | Grid | Sim Time (water+skull) | Attenuation | Focal Shift |
|-----------|------|----------------------|-------------|-------------|
| 4mm | 52x64x64 (213K vox) | 13.0s | **94.3%** | 0.0mm |
| 2mm | 104x128x128 (1.7M vox) | 11.9s | -7.6%* | 0.0mm |

*2mm attenuation anomaly due to target/source positioning in different tissue context at higher resolution — needs investigation.

**3D 16-Element Delay Optimization (4mm grid):**
- **4.7x** focal pressure improvement in 20 iterations (2.6s/iter)
- Loss: -1.66e-5 to -7.79e-5
- Delay pattern shows clear alternating phase structure (~0ns vs ~1200ns)

### 256-Element FWI on SimNIBS Ernie Head (A100, 2026-04-05)

Full waveform inversion on the MNI152 head from SimNIBS with 256-element helmet array. Run with: `modal run scripts/modal_fwi_256.py`

| Step | Result |
|------|--------|
| Head model | Ernie MNI152 (182x238x282 at 1mm, 10 tissues → 6 remapped) |
| Array | 256 elements, semi-ellipsoidal Fibonacci spiral (Oxford-UCL geometry) |
| Forward sim (555kHz, 2D slice) | 7.5s, p_max=0.806 Pa |
| FWI (10 iters, optax Adam) | 2.6s/iter, c_range [1498, 1506] m/s |
| SCICO integration | PDHG + IsotropicTVNorm ready (needs flax dep fix) |

Next: add flax to deps, tune FWI learning rate, scale source amplitude for realistic ARFI displacement.

### Helmholtz (Frequency-Domain) vs Time-Domain

| Solver | p_max | Correlation |
|--------|-------|-------------|
| Time-domain (PSTD) | 0.311 | — |
| Helmholtz (GMRES) | 0.287 | **0.82** |

Good agreement between solvers on homogeneous water. Helmholtz is faster for single-frequency steady-state problems.

## Device Range

This simulation platform is **device-agnostic**, covering the full range of TUS hardware:

| Tier | Elements | Example Devices | Status |
|------|----------|----------------|--------|
| Entry | 1-4 | NeuroFUS CTX-500, Sonic Concepts | Supported |
| Mid-range | 16-32 | **OpenLIFU** (Openwater Health) | **Primary target** — full optimization pipeline |
| High-end | 128-256 | **Oxford-UCL 256-element helmet** (Martin, Stagg, Treeby 2025) | Scalable architecture |
| Research | Custom | BabelBrain-supported, Insightec ExAblate | Any geometry via array API |

Reference: Martin et al. "Ultrasound system for precise neuromodulation of human deep brain circuits." *Nature Communications* 16:8024 (2025).

## Vision: Multimodal MRI-TFUS-ARFI Simulation

The long-term goal is a **multimodal simulation chain** connecting acoustic simulation with MRI brain modeling:

```
j-Wave (acoustic)  →  F = 2αI/c (radiation force)  →  µ∇²u + F = ρü (elastodynamics)  →  MRI Bloch sim (ARFI phase maps)
```

When TFUS displaces brain tissue by 1-10µm, MRI detects this via motion-sensitizing gradients (MR-ARFI). Our existing components — j-Wave for acoustics, sbi4dwi's BrainMaterialMap for tissue elastography, and neurojax's BEM/Bloch simulation — form the building blocks. The missing link is coupling the displacement field to KomaMRI spin positions for synthetic MR-ARFI images.

**Key enabling science:**
- **Kuhl CANN** (Linka, St Pierre, Kuhl 2023): Region-specific brain shear moduli — corpus callosum displaces 3.4x more than cortex from same radiation force. Bayesian CANN (2025) provides uncertainty distributions.
- **Butts Pauly MR-ARFI** (Kaye & Pauly 2011/2013): GRE + bipolar motion-sensitizing gradients (MSGs), ~0.1µm sensitivity, ~100ms/image. MSGs are gradient lobe pairs that encode tissue displacement into MRI phase — same principle as diffusion encoding but for deterministic ultrasound-induced displacement. PRF thermometry for temperature monitoring.
- **KomaMRI**: Best MRI simulator for ARFI — Pulseq-compatible, GPU-accelerated, supports time-dependent displacement via Motion objects. POSSUM lacks MSG support; JEMRIS capable but slow.

This closes the loop between MRI monitoring and acoustic simulation in the Oxford-UCL interleaved MRI-TFUS paradigm.

## Context

- **NeuroTechX Global NeuroHack** — hackathon project
- **Openwater Health** — collaboration proposal (mid-range devices)
- **Oxford-UCL MRI-TFUS** — target for high-end simulation (Martin, Stagg, Treeby)
- **SCI Institute head model** — test dataset (Warner et al. 2019, CC-BY 4.0)
- **Modal A100** — cloud GPU for simulation and optimization
- **DGX Spark** — local GPU compute target

### 3D FWI on brain-fwi (A100, 2026-04-06)

Full 3D Full Waveform Inversion using brain-fwi project with 256-element helmet array.
Run with: `cd ~/dev/brain-fwi && modal run scripts/modal_mida_256.py`

| Step | Result |
|------|--------|
| Head model | Synthetic 96^3 at 2mm (MIDA pending upload) |
| Array | 256 elements, Fibonacci helmet with face exclusion |
| Forward sim | **10.9s** on A100, p_max = 0.788 Pa |
| Observed data | 4 shots × 16 sensors = (4, 234, 16), 8.6s |
| FWI (5 iters, 500kHz) | **50.3s** (10s/iter), loss = 0.081 |
| Velocity recovery | [1411, 2166] m/s from initial 1500 (skull = 2800 true) |

Uses brain-fwi's reparameterized velocity (sigmoid bounds), multi-frequency banding, and gradient smoothing.

### Multi-Frequency 3D FWI (A100, 2026-04-06)

3-band FWI (50→200→500 kHz) with 256-element helmet, 8 sources × 32 sensors.
Run with: `cd ~/dev/brain-fwi && modal run scripts/modal_extended_fwi.py`

| Band | Frequency | Loss | Velocity |
|------|-----------|------|----------|
| 1 (coarse) | 50-150 kHz | 0.187→0.080 | 1413-2231 m/s |
| 2 (medium) | 150-350 kHz | 0.186→0.091 | 1402-3762 m/s |
| 3 (fine) | 350-600 kHz | 0.132→0.097 | 1402-**4312** m/s |

**Skull velocity recovered to 4312 m/s** (true=4080). Brain MSE=2776. Total: 222s on A100.
