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

Full pipeline on SCI Institute head model (Warner et al. 2019), 208x256x256 at 1mm, 8 tissue types:

| Step | Time | Result |
|------|------|--------|
| Load segmentation (NRRD) | <1s | 8 tissues, 13.6M labeled voxels |
| Map acoustic properties | <1s | c: 1500-4080 m/s, rho: 1000-1900 kg/m^3 |
| j-Wave water simulation (2D axial) | 9.3s | p_target = 2.13e-4 Pa |
| j-Wave skull simulation (heterogeneous) | 2.6s | p_target = 1.5e-5 Pa |
| **Skull attenuation** | | **93%** (matches literature for cortical bone at 400kHz) |
| Delay optimization (10 iters, Adam) | 22.0s | 2.2s/iter on A100 |
| **Focal pressure improvement** | | **15x** via gradient-optimized delays |
| Optimized delays | | [-843ns, +70ns] for 2-element array |

Run with: `modal run scripts/modal_tus_pipeline.py`

## Context

- **NeuroTechX Global NeuroHack** — hackathon project
- **Openwater Health** — collaboration proposal
- **SCI Institute head model** — test dataset (Warner et al. 2019, CC-BY 4.0)
- **Modal A100** — cloud GPU for simulation and optimization
- **DGX Spark** — local GPU compute target
