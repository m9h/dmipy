
# Dmipy-JAX: Differentiable Microstructure Imaging

> [!NOTE]
> This is a JAX-accelerated port of the original **Dmipy** toolbox. It leverages **Equinox**, **Optimistix**, and **JAX** for GPU-accelerated, differentiable microstructure modeling.

**Dmipy-JAX** is designed for the **reproducible estimation of diffusion MRI-based microstructure features**. It maintains the modular philosophy of the original Dmipy but rebuilds the core engine for modern AI/ML workflows.

## Key Features

*   **JAX-Based**: Fully differentiable end-to-end. Compute gradients of your signal models with respect to any parameter.
*   **GPU Acceleration**: Run fitting and simulations on GPUs for massive speedups (order of magnitude faster than CPU).
*   **Kidger Stack Integration**: Built using `Equinox` for safe model building and `Optimistix` for robust non-linear least squares optimization.
*   **Legacy Compatibility**: Maintains the friendly, composable API of the original Dmipy where possible.

## Feature Parity

| Feature / Model | dmipy (Original) | dmipy-jax (New) | Notes |
| :--- | :---: | :---: | :--- |
| **Infrastructure** | | | |
| Acquisition Class | ✅ | ✅ | `JaxAcquisition` (Simplified) |
| Model Composition | ✅ | ✅ | `compose_models` |
| Fitting (Voxelwise) | ✅ | ✅ | `fit_voxel` (optimistix) |
| **Cylinder Models** | | | |
| Stick (C1) | ✅ | ✅ | `C1Stick` |
| Cylinder (C2) | ✅ | ✅ | `RestrictedCylinder` (Soderman), `CallaghanRestrictedCylinder` |
| Zeppelins | ✅ | ✅ | `G2Zeppelin`, `TortuosityModel` |
| **Gaussian Models** | | | |
| Ball (G1) | ✅ | ✅ | `G1Ball` |
| Tensor (G2) | ✅ | ✅ | `Tensor` (Full Ellipsoid) |
| **Sphere Models** | | | |
| Sphere (S2) | ✅ | ✅ | `SphereStejskalTanner`, `SphereGPD` (SANDI), `SphereCallaghan` |
| **Plane Models** | | | |
| Planes | ✅ | ✅ | `PlaneStejskalTanner`, `PlaneCallaghan` |

## Additional Features

These features are new to **dmipy-jax** and were not present in the original library:

*   **MCMC Inference**: Full Bayesian inference using the No-U-Turn Sampler (NUTS) via `Blackjax`.
*   **Simulation (Monte Carlo)**: High-performance, GPU-accelerated Monte Carlo simulation of diffusion in complex geometries (`dmipy_jax.simulation`).
*   **Tractography**: A Differentiable Streamline Integrator that allows gradients to flow through tractography steps for end-to-end optimization.

## Examples & Tutorials

We are actively porting the original Dmipy tutorials to JAX. 

### ✅ Verified JAX Examples

These scripts demonstrate the new differentiable path and GPU acceleration:

*   **[Simulate NODDI/SANDI](examples/simulate_noddi_sandi_jax.py)**: Forward simulation of complex multi-compartment models (Stick + Zeppelin + Ball) on the GPU.
*   **[AxCaliber Synthetic Fit](examples/example_axcaliber_synthetic.py)**: Fitting Gamma-distributed cylinder radii (`RestrictedCylinder`) to synthetic data.
*   **[Real Data Analysis](examples/demo_chap_real_data.py)**: Fitting models to real DWI data (requires dataset).

### 🚧 Legacy Notebooks

The `.ipynb` notebooks in `examples/` are from the original Dmipy library. They are currently being ported to use `dmipy_jax` and may not run out-of-the-box.

## Installation

This project uses **`uv`** for modern Python dependency management.

```bash
# Clone the repository
git clone https://github.com/m9h/dmipy.git
cd dmipy

# Sync dependencies
uv sync
```

To run checks and tests:

```bash
# Check environment
uv run python check_env.py

# Run tests
uv run pytest
```

## Original Dmipy

This project is a fork of the excellent **Dmipy** toolbox.

**Original Documentation**: http://dmipy.readthedocs.io/
**Original Repository**: https://github.com/AthenaEPI/dmipy

**Original Description**:
The Dmipy software package facilitates the reproducible estimation of diffusion MRI-based microstructure features. It does this by taking a completely modular approach to Microstructure Imaging. Using Dmipy you can design, fit, and recover the parameters of any multi-compartment microstructure model in usually less than 10 lines of code.

### Citation

If you use this software, please cite the original Dmipy paper:

*   **Primary Reference**: Rutger Fick, Demian Wassermann and Rachid Deriche, "The Dmipy Toolbox: Diffusion MRI Multi-Compartment Modeling and Microstructure Recovery Made Easy", *Frontiers in Neuroinformatics* 13 (2019): 64.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.
Copyright (c) 2017 Rutger Fick & Demian Wassermann
Copyright (c) 2024-2025 Morgan Hough
