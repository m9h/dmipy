# Project Roadmap

## Phase 1: Foundation
- [x] **JAX Port Core**:
    - [x] Port `Acquisition` class to `JaxAcquisition` (PyTree registered).
    - [x] Port core Signal Models (Stick, Ball, Sphere).
    - [x] Implement `compose_models` for multi-compartment modeling.
- [ ] **Optimization Engine V1**:
    - [ ] Integrate **optimistix** for Levenberg-Marquardt (LM) solvers.
        - *Rationale*: LM is the gold standard for NLLS in MRI, offering 2nd-order convergence and NaN-safe Trust Regions.
    - [ ] Implement voxel-wise fitting using `optimistix.least_squares`.

## Phase 2: Optimization
- [ ] **Advanced Fitting**:
    - [ ] Scaling to 1M+ voxels using `jax.vmap`.
    - [ ] Implement Microstructure Fitting (Deterministic) using `optimistix`.
    - [ ] Prepare for Neural Network Training (Stochastic) using `optax`.
- [ ] **Verification**:
    - [ ] Benchmark against original `dmipy`.
    - [ ] Validate accuracy on synthetic phantom data.
