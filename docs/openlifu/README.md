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

## Context

- **NeuroTechX Global NeuroHack** — hackathon project
- **Openwater Health** — collaboration proposal
- **SCI Institute head model** — test dataset (Warner et al. 2019)
- **DGX Spark** — GPU compute target for simulation and optimization
