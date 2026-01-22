# Prompt: Develop MDT Docker Container & Adapter Layer

**Objective**: Create a functioning Docker container for the [Microstructure Diffusion Toolbox (MDT)](https://github.com/Robbert-Harms/MDT) and implement a Python "Adapter Layer" to use it as a test oracle for `dmipy-jax`.

**Context**:
MDT is a reference implementation for microstructure modeling. It requires OpenCL and has a complex model instantiation API. We need a containerized environment to run it reliably and a simplified Python interface to generate ground truth signals.

## Task 1: Build MDT Docker Container (OpenCL Support)

1.  **Create/Locate Dockerfile**:
    *   Reference MDT's official `containers/Dockerfile.intel` (for CPU OpenCL).
    *   Ensure the base image supports Python 3 and OpenCL (e.g., `ubuntu:22.04` with `intel-opencl-icd`).
    *   Install system dependencies: `ocl-icd-libopencl1`, `opencl-headers`, `clinfo`, `git`.
    *   Install MDT: `pip install mdt`.

2.  **Build & Verify**:
    *   Build tag: `mdt-oracle:latest`.
    *   Verify OpenCL: Run `docker run --rm mdt-oracle clinfo` to ensure a generic CPU device (e.g., "Intel(R) CPU Runtime for OpenCL") is visible.

## Task 2: Implement MDT Adapter Layer (`mdt_adapter.py`)

Create a Python script to be run *inside* the container that abstracts away MDT's complex API.

**Requirements**:
1.  **Model Instantiation**: Implement a factory function `get_mdt_model(name: str)`.
    *   *Challenge*: `mdt.get_model("BallStick")` may fail.
    *   *Solution*: You likely need to construct the model using model templates or composite strings.
    *   *Example*: For "Stick", you might need something like `mdt.get_component('composite_models', 'Stick')` or define a custom composite string `w * Stick`.
2.  **Synthesize Signal**: Implement `simulate_signal(model, bvals, bvecs, params)`.
    *   Create a minimal protocol object in MDT format.
    *   Use MDT's simulation or forward-pass methods to generate signal $E(b,n)$.

**Reference Models to Implementation**:
*   `Stick` (Gaussian Phase Distribution, zero radius)
*   `Ball` (Isotropic Gaussian)
*   `Zeppelin` (Cylindrically symmetric Gaussian)
*   `NODDI` (Watson-dispersed Sticks + Ball + Zeppelin)

## Task 3: Verification Script (`verify_oracle.py`)

Create a script that:
1.  Defines ground truth parameters (e.g., diffusivity $D=2.0 \mu m^2/ms$).
2.  Simulates signal using your `mdt_adapter.py` inside the container.
3.  Prints the output signal for validation against analytical computations.

## Deliverables
1.  `docker/Dockerfile.mdt`.
2.  `docker/mdt_adapter.py`.
3.  `docker/build_and_run.sh` script to build image and run verification.
