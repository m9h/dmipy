"""
SimulationLibrary: HDF5-backed storage for parameter–signal dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class SimulationLibrary:
    """In-memory representation of a simulation dictionary.

    Attributes
    ----------
    params : jnp.ndarray
        ``(N, P)`` parameter array.
    signals : jnp.ndarray
        ``(N, M)`` signal array.
    parameter_names : list of str
        Ordered names matching columns of ``params``.
    metadata : dict
        Arbitrary metadata (model name, acquisition info, etc.).
    """

    params: jnp.ndarray
    signals: jnp.ndarray
    parameter_names: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def n_entries(self) -> int:
        return self.params.shape[0]

    @property
    def theta_dim(self) -> int:
        return self.params.shape[1]

    @property
    def signal_dim(self) -> int:
        return self.signals.shape[1]

    # ------------------------------------------------------------------ #
    # HDF5 I/O
    # ------------------------------------------------------------------ #

    def save_hdf5(self, path: str) -> None:
        """Write library to an HDF5 file."""
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(path), "w") as f:
            f.attrs["schema_version"] = 1
            f.create_dataset("params", data=np.asarray(self.params))
            f.create_dataset("signals", data=np.asarray(self.signals))
            f.attrs["parameter_names"] = self.parameter_names
            for k, v in self.metadata.items():
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = str(v)

    @classmethod
    def load_hdf5(cls, path: str, mmap: bool = False) -> "SimulationLibrary":
        """Load library from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to ``.h5`` file.
        mmap : bool
            If *True*, open the file with ``driver='core', backing_store=False``
            which pre-loads the entire file into a RAM buffer for faster
            sequential reads.  Despite the parameter name this is **not**
            true memory-mapped I/O; the full dataset is still resident in RAM.
        """
        import h5py

        kwargs = {}
        if mmap:
            kwargs = dict(driver="core", backing_store=False)

        with h5py.File(str(path), "r", **kwargs) as f:
            if "schema_version" not in f.attrs:
                import warnings
                warnings.warn(
                    f"HDF5 file {path} has no schema_version attribute. "
                    "It may have been created by an older version of SBI4DWI.",
                    stacklevel=2,
                )
            params = jnp.array(f["params"][:])
            signals = jnp.array(f["signals"][:])
            parameter_names = list(f.attrs.get("parameter_names", []))
            metadata = {
                k: v for k, v in f.attrs.items()
                if k not in ("parameter_names", "schema_version")
            }

        return cls(
            params=params,
            signals=signals,
            parameter_names=parameter_names,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Normalisation helpers
    # ------------------------------------------------------------------ #

    def normalise_signals(self) -> "SimulationLibrary":
        """L2-normalise each signal row (in-place-ish, returns self)."""
        norms = jnp.linalg.norm(self.signals, axis=1, keepdims=True)
        norms = jnp.maximum(norms, 1e-12)
        self.signals = self.signals / norms
        return self
