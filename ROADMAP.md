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
    - [ ] **Implement `JaxMultiCompartmentModel` Wrapper**: Create a high-level class wrapping `compose_models` to handle dictionary-based parameter management and provide a `fit()` API compatible with legacy `dmipy`.
    - [ ] **Implement Global Initialization**: Port the "Brute" phase of `Brute2FineOptimizer` using `jax.vmap` grid search to provide robust initial guesses for the LM solver.
    - [ ] **Port Missing Models**: Implement `TortuosityModel`, `RestrictedCylinder` classes, and other specialized models used in examples.
- [ ] **Advanced Fitting**:
    - [ ] Scaling to 1M+ voxels using `jax.vmap`.
    - [ ] Implement Microstructure Fitting (Deterministic) using `optimistix`.
    - [ ] Prepare for Neural Network Training (Stochastic) using `optax`.
- [ ] **Verification**:
    - [ ] Benchmark against original `dmipy`.
    - [ ] Validate accuracy on synthetic phantom data.
    - [ ] Port key examples (`ball_and_stick`, `noddi`) to verify end-to-end user experience.
