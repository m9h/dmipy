
# Track A: Official uGUIDE Benchmark Instructions

This document provides instructions to benchmark the official `uGUIDE` framework against the `dmipy-jax` native implementation.

## Prerequisites

1.  **Install uGUIDE**:
    Ensure you have a Python environment with `uGUIDE` installed.
    ```bash
    pip install uguide
    # Or clone from https://github.com/uGUIDE-Toolbox/uGUIDE and install
    ```

2.  **Data**:
    The training data has been generated in `uGUIDE_train_data.h5`.
    The test data is in `uGUIDE_test_data.h5`.

## Training (Track A)

Create a script `train_uguide.py` to train the uGUIDE model using the generated HDF5 data.

```python
import uguide
import h5py
import numpy as np
import pickle

# 1. Load Data
with h5py.File('uGUIDE_train_data.h5', 'r') as f:
    signals = f['signals'][:]
    params = f['parameters'][:]

# 2. Train uGUIDE
# uGUIDE typically infers the posterior given the data.
# Note: Adjust the API calls below based on the specific version of uGUIDE you are using.
# This assumes a standard fit(theta, x) API.

# Normalize if uGUIDE doesn't handle it automatically (it usually does or expects it).
# For fairness, we use the same data.

print("Training uGUIDE...")
# Example API usage (pseudo-code, please verify with uGUIDE docs):
model = uguide.train(
    theta=params,
    x=signals,
    prior=None, # or defined prior
    model_type='mdn', # or flow
    n_epochs=20
)

# 3. Save Model
model.save('uguide_model.pkl')
print("Saved uGUIDE model.")
```

## Inference & Evaluation

To evaluate and compare with JAX:

1.  **Run Inference on Test Set**:
    Load `uGUIDE_test_data.h5` and use the trained `uguide_model` to predict parameters.

2.  **Save Predictions**:
    Save the predicted parameters (means) to `uguide_predictions.npy`.

3.  **Run Benchmark Comparison**:
    Run the provided `examples/sbi/benchmark_comparison.py`.
    It will automatically look for `uguide_predictions.npy` and include it in the report if found.

    ```bash
    python examples/sbi/benchmark_comparison.py --uguide_results uguide_predictions.npy
    ```

## Notes

-   **Performance**: Measure the time taken for the inference step (samples/sec) and provide it to the benchmark script or note it manually.
-   **Hardware**: Ensure both tracks are run on the same hardware (CPU vs GPU) for fair comparison.
