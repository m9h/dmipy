# MDT Docker Oracle

This directory contains the Docker configuration and adapter layer for using the **Microstructure Diffusion Toolbox (MDT)** as a "Ground Truth" Test Oracle for benchmarking `dmipy-jax`.

## Overview

The `mdt-oracle` container provides a standardized environment with:
- **MDT 1.2.7**: Installed from PyPI.
- **OpenCL Runtime**: Configured with `pocl` (Portable Computing Language) for cross-platform hardware acceleration (works on x86 and ARM64/Apple Silicon).
- **Python Adapter**: A simplified interface (`mdt_adapter.py`) to instantiate models and simulate signals without managing complex MDT protocols manually.

## Files

- `Dockerfile.mdt`: The container definition.
- `mdt_adapter.py`: Python module bridging `dmipy-jax` parameters to MDT's API.
- `verify_oracle.py`: Script to verify the installation and signal simulation accuracy.
- `build_and_run.sh`: Helper script to build and verify the container.

## Usage

### 1. Build and Verify
The easiest way to get started is to use the provided shell script:

```bash
./build_and_run.sh
```

This will:
1. Build the docker image `mdt-oracle:latest`.
2. Run a temporary container executing `verify_oracle.py`.
3. Print the simulated signals for `Stick` and `Ball` models.

### 2. Manual Build
```bash
docker build -f Dockerfile.mdt -t mdt-oracle:latest .
```

### 3. Using the Adapter
The `mdt_adapter.py` is designed to be imported within the container. It provides two main functions:

#### `get_mdt_model(name: str)`
Instantiates an MDT model class.
- Supported: `'Stick'`, `'Ball'`, `'Zeppelin'`, `'NODDI'`.
- *Note*: `Stick`, `Ball`, and `Zeppelin` are implemented via MDT's `Tensor` model with specific constraints.

#### `simulate_signal(model, bvals, bvecs, params)`
Generates synthetic signals.
- **`model`**: Instance returned by `get_mdt_model`.
- **`bvals`**: 1D array of b-values in SI units ($s/m^2$).
    - *Note*: The adapter automatically converts these to $s/mm^2$ for MDT protocol file compatibility.
- **`bvecs`**: $N \times 3$ array of gradient vectors.
- **`params`**: Dictionary of model parameters (e.g., `{'Tensor.d': 2e-9}`).

**Example:**
```python
import mdt_adapter
import numpy as np

# 1. Get Model
stick = mdt_adapter.get_mdt_model('Stick')

# 2. Define Protocol (SI Units)
bvals = np.array([0, 1e9, 2e9])  # s/m^2
bvecs = np.array([[0,0,0], [1,0,0], [0,1,0]]) # x, y

# 3. Define Parameters
params = {
    'Tensor.d': 2e-9,      # Diffusivity
    'Tensor.theta': 0.0,   # Orientation
    'Tensor.phi': 0.0,
    'S0.s0': 1.0           # Signal at b=0
}

# 4. Simulate
signal = mdt_adapter.simulate_signal(stick, bvals, bvecs, params)
print(signal)
```

## Troubleshooting

- **OpenCL Errors**: Run `clinfo` inside the container to verify the OpenCL platform is detected. The Dockerfile installs `pocl-opencl-icd` which should work on most CPUs.
- **Volume Dropping**: If MDT drops volumes (logs "Using X out of Y volumes"), check that your b-values are not effectively zero relative to the expected unit. The adapter expects $s/m^2$ (e.g., 1000 -> $10^9$) and handles the conversion.
- **Python 3.10+**: The adapter includes a monkeypatch for `collections.Mapping` to support newer Python versions with MDT 1.2.7.
