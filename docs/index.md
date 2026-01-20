# Welcome to dmipy-jax

**dmipy-jax** is a high-performance port of the Diffusion Microstructure Imaging in Python (Dmipy) library, leveraging [JAX](https://github.com/google/jax) for GPU acceleration, automatic differentiation, and just-in-time compilation.

## Key Features
- **GPU Acceleration**: Fit generic multi-compartment models on 1,000,000+ voxels in seconds.
- **Auto-Differentiation**: Gradients are computed automatically, enabling efficient optimization.
- **Modularity**: Compose models like Legos, same as original Dmipy.

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

tutorials/first_steps
tutorials/complex_synthetic_data
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/modules
```

## Indices and tables
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
