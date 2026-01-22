
You are an expert DevOps and ML Engineer. Your task is to create a `Dockerfile` and a `README.md` to build a containerized "Test Oracle" environment for Diffusion MRI Simulation-Based Inference (SBI).

## Background
We need to reproduce the data simulations from the paper *"Uncertainty mapping and probabilistic tractography using Simulation-based Inference in diffusion MRI"* (bioRxiv 2024). The code is hosted at `https://github.com/SPMIC-UoN/SBI_dMRI`.

## Requirements

### 1. Base Image
- Use an **NVIDIA PyTorch** base image to ensure GPU acceleration.
- Recommended: `nvcr.io/nvidia/pytorch:23.10-py3` (or a compatible version with Python 3.10+).

### 2. Dependencies
- **SBI Toolbox**: Install `sbi==0.22.0` (as specified in the paper).
- **dMRI Tools**: Install `dipy` and `nibabel`.
- **Repo**: Clone the repository `https://github.com/SPMIC-UoN/SBI_dMRI` into `/app/SBI_dMRI`.

### 3. Simulation Scripts
The paper generates synthetic data using the **Ball & Sticks** model with specific acquisition protocols (UKBiobank-like).
- Identify the simulation script in the cloned repo (likely in a `simulations/` or `data_generation/` folder).
- If no explicit script exists, create a Python script `/app/generate_oracle_data.py` that:
    1.  Imports the simulation utilities from `SBI_dMRI`.
    2.  Defines a "Ball & Stick" model (diffusivity=1.7e-3 mm²/s, etc.).
    3.  Simulates signal attenuation for b=1000 and b=2000 s/mm² (100 directions).
    4.  Saves the output (signals and parameters) to `/data/oracle_sims.npz`.

### 4. Deliverables
Produce the following files:
1.  **`Dockerfile`**: The complete definition.
2.  **`build_and_run.sh`**: A bash script to build the image and run the simulation, mounting a local directory to `/data` to capture outputs.

## Constraints
- Ensure `fsl` is NOT required unless absolutely necessary (keep image light).
- The environment must be reproducible.
