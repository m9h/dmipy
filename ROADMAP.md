# Project Roadmap

## Phase 1: Foundation
- [x] **JAX Port Core**:
    - [x] Port `Acquisition` class to `JaxAcquisition` (PyTree registered).
    - [x] Port core Signal Models (Stick, Ball, Sphere).
    - [x] Implement `compose_models` for multi-compartment modeling.
- [x] **Optimization Engine V1**:
    - [x] Integrate **optimistix** for Levenberg-Marquardt (LM) solvers.
        - *Rationale*: LM is the gold standard for NLLS in MRI, offering 2nd-order convergence and NaN-safe Trust Regions.
    - [x] Implement voxel-wise fitting using `optimistix.least_squares`.

## Phase 2: Optimization & Compatibility
- [ ] **JAX-Native Modeling Framework**:
    - [x] **Implement `JaxMultiCompartmentModel` Wrapper**: Create a high-level class wrapping `compose_models` to handle dictionary-based parameter management and provide a `fit()` API compatible with legacy `dmipy`.
    - [ ] **Implement Global Initialization**: Port the "Brute" phase of `Brute2FineOptimizer` using `jax.vmap` grid search to provide robust initial guesses for the LM solver.
    - [ ] **Port Missing Models**: Implement `TortuosityModel`, `RestrictedCylinder` classes, and other specialized models used in examples.
- [ ] **Advanced Fitting & UQ (New)**:
    - [x] **Surrogate Modeling**: Implemented Generalized Polynomial Chaos (gPC) for accelerating slow models (`dmipy_jax.core.surrogate`).
    - [x] **Uncertainty Quantification**: Implemented Tier 1 (CRLB) uncertainty estimation in `JaxMultiCompartmentModel`.
    - [ ] Scaling to 1M+ voxels using `jax.vmap`.
    - [ ] Prepare for Neural Network Training (Stochastic) using `optax`.
- [ ] **Verification**:
    - [ ] Benchmark against original `dmipy`.
    - [ ] Validate accuracy on synthetic phantom data.
    - [ ] Port key examples (`ball_and_stick`, `noddi`) to verify end-to-end user experience.

## Phase 3: Inverse & Global Fitting (scico)
- [x] **Inverse Architect Foundation**:
    - [x] Integrate **scico** dependency.
    - [x] Implement `MicrostructureOperator` to wrap `dmipy` models as JAX-compatible operators.
- [ ] **Linearized Inference (AMICO)**:
    - [x] Port core AMICO solver using ADMM (`AMICOSolver`).
    - [ ] Build robust dictionary generators to replace legacy `cvxpy` implementation.
    - [ ] Benchmark `scico` vs `cvxpy` for speed and accuracy.
- [ ] **Global Reconstruction**:
    - [x] Implement `GlobalOptimizer` with Total Variation (TV) regularization.
    - [ ] Calibrate regularizers for in-vivo data.
    - [ ] Investigate advanced priors (Hessian, Non-local Means).
- [ ] **Advanced Features**:
    - [ ] Constrained Spherical Deconvolution (CSD) via sparse reconstruction.
    - [ ] Joint reconstruction and parameter estimation.

## PyPulseq Integration
- **Status:** In Progress
- **Goal:** Bridge the gap between sequence design (PyPulseq) and high-performance simulation (Dmipy-JAX).
- **Features:** 
    - `dmipy_jax.external.pulseq`: Bridge module to convert `.seq` files or objects into JAX-compatible structures.
    - Support for `GeneralSequence` (dense waveforms) for Bloch simulation.
    - Support for `JaxAcquisition` (b-vals/b-vecs) for microstructure fitting.

## Phase 4: Beyond Parity (Advanced Spherical Models)
- **Goal:** Extend the biophysical modeling capabilities beyond the original `dmipy` scope, focusing on advanced spherical and exchange models.
- **Key Models:**
    - [ ] **Permeable Spheres (Exchange)**: Implement models with finite membrane permeability to account for intra/extra-cellular exchange (e.g., Kärger, NEXI).
        - *Rationale*: Crucial for long diffusion times and cell membrane characterization.
    - [ ] **Concentric Spheres**: Model soma with nucleus (two compartments).
        - *Rationale*: Explains relaxation-diffusion correlations (T1/T2 vs D) in soma imaging.
    - [ ] **Restricted Ellipsoids**: Analytical or numerical approximations for non-spherical somas.
        - *Rationale*: Captures microscopic anisotropy of gray matter cell bodies.
    - [ ] **Gamma-Distributed Spheres**: Native optimization of diameter distributions (polydispersity).
        - *Rationale*: More realistic than single-diameter models; leverages JAX for distribution parameter optimization.
    - [ ] **T2-Dot**: Stationary compartment with T2 relaxation.
        - *Rationale*: Modeling myelin water or other trapped, non-diffusing pools with distinct relaxation times.
