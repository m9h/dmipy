"""
dmipy-jax: A JAX-accelerated port of the Diffusion Microstructure Imaging in Python (Dmipy) library.

This package provides GPU-accelerated signal models, fitting algorithms, and tools for 
microstructure imaging. It mirrors the structure of the original dmipy library but leverages
JAX for automatic differentiation, vectorization, and compilation.

Submodules
----------
- `acquisition`: Simplified acquisition scheme handling for JAX.
- `cylinder`: Orientation-dispersed cylinder models (e.g., Stick).
- `gaussian`: Gaussian diffusion models (e.g., Ball).
- `sphere`: Restricted diffusion in spheres.
- `fitting`: Generic voxel-wise fitting using JAXopt.
- `composer`: Utilities for composing multi-compartment models.
- `inference.mcmc`: Bayesian inference using Blackjax (NUTS).
"""

from dmipy_jax import acquisition
from dmipy_jax import cylinder
from dmipy_jax import gaussian
from dmipy_jax import sphere
from dmipy_jax import fitting
from dmipy_jax import inverse
from dmipy_jax import composer

__all__ = [
    'acquisition',
    'cylinder',
    'gaussian',
    'sphere',
    'fitting',
    'inverse',
    'composer',
]
