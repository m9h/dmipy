# Project Roadmap V4: Clinical Translation & Real-World Validation

**Objective**: Transcend analytical limits using Differentiable Programming (JAX), moving from **Theoretical Simulation-Based Inference (SBI)** to **Real-World Clinical Validation**.

---

## 🏗️ Completed Foundations (Phases 1-5)

Everything from `dmipy` porting to the latest SIREN networks.

- [x] **Core Infrastructure**: `dmipy-jax`, `JaxAcquisition`, `equinox`/`diffrax` stack.
- [x] **Differentiable Physics**: Bloch Simulator, Particle Engine, Differentiable SDEs.
- [x] **Optimization & OED**: CRB Minimization, Bayesian OED, Algebraic Protocol Design.
- [x] **Simulation-Based Inference (SBI)**:
    -   Amortized Posterior Estimation (Normalizing Flows).
    -   Physics-informed encoders and algebraic initializers.
    -   Hardware acceleration (TPU/GPU).
- [x] **Advanced Biophysics**:
    -   Neural Exchange (Neural ODEs).
    -   **SIREN Networks**: Continuous signal representation (implemented in `dmipy_jax/core/networks.py`).

---

## 🏥 Phase 6: Real-World Validation (ACTIVE)

*Goal: Validate computational models against biological ground truth.*

- [ ] **Differentiable Histology Bridge**:
    -   **Dataset**: [Histo-µSim](https://zenodo.org/records/14559356) (Histology-informed microstructural diffusion simulations).
    -   **Task**: Create `dmipy_jax/validation/histology.py`.
    -   **Validation Loop**: Differentiate fitting error w.r.t histology-predicted signals to validate microstructural indices (e.g., cell radius vs. actual radius).

- [ ] **Phanthom Validation**:
    -   Validate SBI models against physical phantoms (e.g., MGH/Fibercup) where ground truth is known.

---

## 🧠 Phase 7: Clinical Translation (NEW)

*Goal: Deploy Differentiable MRI in neurosurgical planning.*

- [ ] **Neurosurgical Diffusion Validation**:
    -   **Target Dataset**: **OpenNeuro BTC_preop (ds001226)** and **BTC_postop** (Brain Tumor Center, gliomas).
    -   **Context**: Multi-shell HARDI data from glioma patients.
    -   **Objective**: Test if SBI/SIREN models can delineate tumor boundaries better than standard DTI/CSD.
- [ ] **Pathology-Specific Priors**:
    -   Adapt `dmipy_jax` priors for infilatrating tumor tissue (e.g., increased diffusivity, reduced anisotropy).
    -   Implement "Tumor-Aware" SBI networks trained on mixed healthy/pathological simulations.
- [ ] **Clinical Metrics**:
    -   Compare "Time-to-Inference" vs. standard tools (FSL/Mrtrix).
    -   Quantify uncertainty in tumor margin estimation.

---

## 🚀 Next Development Steps (Agent Prompts)

### 1. Differentiable Histology Bridge (Priority 1)
*   **Goal**: Prove biological accuracy using **Histo-µSim**.
*   **Action**: Implement `HistoDataset` loader and `HistologySimulator` in `dmipy_jax/validation/histology.py`.

### 2. Clinical Dataset Pipeline (Priority 2)
*   **Goal**: Ingest real clinical data.
*   **Action**: Create `dmipy_jax/data/openneuro.py` to fetch and preprocess Basic Tumor Center (BTC) datasets from OpenNeuro.
*   **Verification**: Run standard DTI fit on a glioma subject and visualize the tumor leveraging JAX acceleration.

### 3. Multi-Modal Degeneracy Resolution
*   **Context**: Combine Diffusion with T1/T2 maps (available in OpenNeuro datasets) to resolve microstructure ambiguity.
*   **Action**: Implement Joint T1-Diffusion Inversion in `dmipy_jax/core/multimodal.py`.

---

## 📚 Literature & Tech Stack

### 1. Clinical datasets
*   **OpenNeuro ds001226 (BTC_preop)**: High-grade glioma patients with multi-shell dMRI.
*   **Histo-µSim**: Synthetic ground-truth from histology.

### 2. Simulation-Based Inference (SBI)
*   **Ref**: *Simulation-Based Inference at the Theoretical Limit* (Maximilian et al., 2024).

### 3. Differentiable Physics (SIREN)
*   **Ref**: *Implicit Neural Representation of Multi-shell CSD* (Hendriks et al., 2024).
