"""Data loaders for diffusion MRI datasets.

This package provides standardised loaders for reading dMRI data from various
sources and formats into JAX arrays suitable for the SBI pipeline.

Supported data sources:

* :class:`~dmipy_jax.io.bids.BIDSLoader` -- BIDS-compliant datasets
  (via ``pybids``).
* :mod:`~dmipy_jax.io.hcp` -- Human Connectome Project data (via S3/boto3).
* :class:`~dmipy_jax.io.multi_te.MultiTELoader` -- multi-echo-time
  diffusion datasets.
* :class:`~dmipy_jax.io.wand.WANDLoader` -- WAND (CUBRIC) AxCaliber data
  (via GIN).
* :class:`~dmipy_jax.io.mica.MicaMICsLoader` -- MICA-MICs multi-shell
  dataset.
* :mod:`~dmipy_jax.io.lowfield` -- OpenNeuro ultra-low-field brain MRI
  (ds006557).
* :mod:`~dmipy_jax.io.mesh` -- surface mesh IO (OFF, VTK) for FEM
  simulation.
* :mod:`~dmipy_jax.io.datasets` -- convenience loaders for standard
  validation datasets (Stanford HARDI, Sherbrooke 3-shell, BigMac).
* :func:`~dmipy_jax.io.bids.save_bids_derivative` -- save parameter maps
  as BIDS derivatives with JSON sidecars.

All loaders convert b-values to SI units (s/m^2) and gradient directions
to shape ``(N, 3)`` for direct use with :class:`~dmipy_jax.acquisition.JaxAcquisition`.
"""

from .bids import BIDSLoader, save_bids_derivative
