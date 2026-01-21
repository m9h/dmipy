# SBI Implementation Prompts for Dmipy-JAX

These prompts are designed to guide an AI agent in implementing the 4 key microstructural models referenced in the SBIDTI repository (Eggl et al., 2024): **DTI**, **DKI**, **NODDI**, and **AxCaliber**.

## Common Context (Prepend to all prompts)
> You are working on `dmipy-jax`, a differentiable MRI library using JAX, Equinox, and Optimistix.
> We are implementing Simulation-Based Inference (SBI) as a "Phase 5" feature.
> Your goal is to create "Amortized Inference" modules that replace voxel-wise fitting with neural network inversion.
> Use `sbi` (Python library) or simple NF (Normalizing Flow) implementations in JAX (`distrax` or `flowjax` if available, otherwise generic MLP regressors for simple NPE).
>
> **Reference Paper**: *Simulation-Based Inference at the Theoretical Limit* (Eggl et al., 2024).
> **Objective**: Train a Neural Posterior Estimator (NPE) on simulated data to predict posterior distributions of model parameters given dMRI signals.

---

## Task 1: DTI (Diffusion Tensor Imaging)
**Prompt:**
```text
Implement an SBI inference module for the Diffusion Tensor Imaging (DTI) model.

1.  **Simulation**:
    *   Use `dmipy_jax.signal_models.gaussian_models.Tensor` (or `G2Zeppelin` if full tensor not avail) to generate 100,000 synthetic signal vectors.
    *   **Priors**: Sample eigenvalues ($\lambda_1, \lambda_2, \lambda_3$) from standard biological ranges (0.1 - 3.0 um^2/ms). Sample orientation randomly on the sphere.
    *   **Acquisition**: Use a standard single-shell scheme (b=1000, 32 dirs).
    *   **Noise**: Add Rician noise at SNR=30.

2.  **Inference Network (NPE)**:
    *   Create a Neural Posterior Estimator (NPE) using a Masked Autoregressive Flow (MAF) or simply a Mixture Density Network (MDN) if flows are too complex for now.
    *   Inputs: dMRI Signal (noisy).
    *   Outputs: Posterior distribution over FA (Fractional Anisotropy) and MD (Mean Diffusivity). Note: We often predict invariants rather than full tensor components to avoid rotational issues.
    *   Alternatively: Predict the full lower-triangular tensor components, but augment training data with random rotations.

3.  **Output**:
    *   Script: `dmipy_jax/examples/sbi/train_dti.py`
    *   Validation: Plot Predicted vs True FA and MD for a held-out test set.
```

---

## Task 2: DKI (Diffusion Kurtosis Imaging)
**Prompt:**
```text
Implement an SBI inference module for the Diffusion Kurtosis Imaging (DKI) model.

1.  **Simulation**:
    *   Use `dmipy_jax` to simulate non-Gaussian diffusion signals. If a dedicated DKI analytical model isn't present, simulate using a Multi-Compartment Model (Stick + Ball) which naturally exhibits non-Gaussianity/Kurtosis at high b-values.
    *   **Acquisition**: Two-shell scheme (b=1000, 2000; 60 dirs total) is required for Kurtosis.
    *   **Priors**: Vary compartment fractions to induce varying levels of Kurtosis.

2.  **Inference Network**:
    *   Train an NPE to recover the standard DKI metrics: Mean Kurtosis (MK), Axial Kurtosis (AK), and Radial Kurtosis (RK).
    *   Since analytical DKI fitting is often unstable/noise-sensitive, SBI should demonstrate superior robustness.

3.  **Output**:
    *   Script: `dmipy_jax/examples/sbi/train_dki.py`
    *   Validation: Compare SBI estimates of MK against standard linear least squares (LLS) fitting on noisy data.
```

---

## Task 3: NODDI (Neurite Orientation Dispersion and Density Imaging)
**Prompt:**
```text
Implement an SBI inference module for the NODDI model (Stick + Zeppelin + Ball).

1.  **Simulation**:
    *   Use `JaxMultiCompartmentModel([Stick, Zeppelin, Ball])`.
    *   **Priors**:
        *   $f_{intra}$ (Stick): Beta distribution focused on 0.3-0.7.
        *   $f_{iso}$ (Ball): Sparse prior (mostly near 0, some high values for CSF).
        *   $\kappa$ (Dispersion): Watson distribution concentration. Sample log-uniform.
    *   **Acquisition**: Multi-shell (b=700, 2000; optimized for NODDI).

2.  **Inference Network**:
    *   This is the classic "Ambiguous Inverse Problem".
    *   Train an NPE to predict the joint posterior $p(f_{intra}, f_{iso}, \kappa | S)$.
    *   Crucial: Demonstrate that for certain signals, the posterior is multimodal or wide (uncertainty quantification), whereas standard fitting gives a single point estimate.

3.  **Output**:
    *   Script: `dmipy_jax/examples/sbi/train_noddi.py`
    *   Validation: Show posterior corner plots for a "crossing fiber" or "partial volume" voxel.
```

---

## Task 4: AxCaliber / ActiveAx (Axon Diameter Estimation)
**Prompt:**
```text
Implement an SBI inference module for Axon Diameter estimation (AxCaliber or ActiveAx).

1.  **Simulation**:
    *   Use `dmipy_jax.signal_models.cylinder_models.Cylinder` (finite radius).
    *   **Protocol**: Requires High Gradients (Gmax > 300 mT/m) or multiple diffusion times (Delta).
    *   **Priors**: Axon radii $R \in [0.1, 5.0] \mu m$. Gamma distribution parameters if modeling a distribution.

2.  **Inference Network**:
    *   Axon diameter estimation is notoriously difficult and degenerate with noise.
    *   Train an NPE to predict Mean Axon Diameter (MAD).
    *   **theoretical limit**: Reference the "theoretical limit" from the paper – show that for standard gradients (G=40-80 mT/m), the posterior for R < 2um is effectively uniform (uninformative), correctly signaling that the data cannot resolve it.

3.  **Output**:
    *   Script: `dmipy_jax/examples/sbi/train_axcaliber.py`
    *   Validation: Sensitivity analysis plot – Posterior width vs Ground Truth Radius.
```
