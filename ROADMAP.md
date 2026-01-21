# Project Roadmap V3: The Differentiable Science Era

**Objective**: Transcend analytical limits using Differentiable Programming (JAX), focusing on Simulation-Based Inference, Bayesian Optimal Design, and Generative Modeling.

## Completed Foundations (Phases 1-3.5)
- [x] **Core Infrastructure**: JAX port of dmipy signal models, `JaxAcquisition`.
- [x] **Scientific Stack**: Adoption of `equinox`, `diffrax`, `optimistix`, `scico`.
- [x] **Inverse Problems**: Fast AMICO (ADMM), Global TV Regularization.
- [x] **Simulation**: Bloch Simulator (`diffrax`), Particle Engine (`jax-md`), Differentiable SDEs.
- [x] **Basic Inference**: Levenberg-Marquardt fitting (`optimistix`), Variational Inference basics (`inference/variational.py`).

---

## Phase 4: Differentiable Science (COMPLETE)

### 4.1 Optimal Experimental Design (OED)
*Goal: Design the perfect MRI protocol by differentiating through the simulator.*
- [x] **CRB Minimization**: Minimize local uncertainty using Fisher Information (D-Optimality). (**Done in `optimization/acquisition.py`**)
- [x] **Bayesian OED (Next Frontier)**:
    -   Maximize Mutual Information $I(\theta; y) = H(y) - H(y|\theta)$.
    -   Use `MeanFieldGaussian` VI to estimate posterior entropy for diverse tissue priors.
    -   Optimize protocol parameters to minimize average posterior entropy across a population.
- [x] **Algebraic Protocol Design (Advanced)**:
    -   Analyze the polynomial ideal of signal models to find b-values that minimize the Grobner basis complexity. (**Implemented Engine in `algebra/identifiability.py`**)
    -   Goal: "Linearize" the algebra of inversion by selecting optimal shells.

### 4.2 Simulation-Based Inference (SBI)
*Goal: Instant, accurate inversion without analytical approximations.*
- [x] **Physics-Informed Encoders**: Simple MLP inversion.
- [x] **Amortized Posterior Estimation**:
    -   Implement **Normalizing Flows** (Conditioned on signal $y$) to output complex posterior distributions $p(\theta|y)$. (**Implemented RQS Flow in `dmipy_jax/inference/flows.py`**)
    -   Handle multi-modal posteriors (degeneracy) which standard MLPs and VI cannot.
    -   **Self-Supervised Refinement**: Fine-tune the flow on specific patient data using the physics loss (reconstruction error). (**Implemented `SSFTTrainer` in `inference/ssft.py` with Global TV**)
- [x] **Algebraic Initializers**:
    -   Derive lightweight rational function approximations from Elimination Ideals (using Grobner bases). (**Implemented `SymbolicInverter` and DTI Init in `dmipy_jax/fitting/algebraic.py`**)
    -   **SymPy2JAX Acceleration**: Compile these symbolic solutions into TPU-accelerated `eqx.Module` initializers.

### 4.3 Neural Biophysics
*Goal: Discover tissue properties that analytical models ignore.*
- [x] **Neural Exchange**: Learned exchange rates using Neural ODEs. (**Implemented in `biophysics/neural_exchange.py`**)
- [x] **Neural Signal Representations**:
    -   Represent the signal attenuation $E(q)$ as a Convex Neural Network (ICNN) to enforce physical monotonicity/convexity constraints. (**Implemented in `biophysics/neural_signal.py`**)
    -   Learn the "Universal Propagator" from data while guaranteeing $E(0)=1$ and $|q|^2$-like decay.

### 4.4 Uncertainty-Aware Mapping
*Goal: Quantify confidence in every pixel.*
- [x] **Bayesian Inference**: Variational Inference (`MeanFieldGaussian`, `VIMinimizer`) implemented in `inference/variational.py`.
- [x] **Degeneracy Quantification**: Output posterior variance and parameter correlations for every voxel. (**Achieved via VI and Flows**)
- [x] **Exact Algebraic Identifiability**:
    -   **Engine**: Grobner Basis solver implemented in `algebra/identifiability.py`. (**Done**)
    -   **Application**: Analytically solve for degenerate modes (`sympy`) to guide SBI models. (**Implemented in `algebra/wrapper.py` & `optimization/oed.py`**)

### 4.5 Generative Microstructure Phantoms
*Goal: "Infinite data" for amortized inference.*
- [x] **Generative Microstructure Phantoms**:
    -   Combine `diffrax` SDEs with Score-Based Generative Models (SGMs). (**Implemented SGM in `dmipy_jax/simulation/phantoms.py`**)
    -   Learn joint distribution $p(\theta)$ of microstructural parameters.

---

## Phase 5: Validation & Clinical Translation (ACTIVE)
- [ ] **Differentiable Histology Bridge**: Direct optimization of MRI models against histology inputs.
- [ ] **Multi-Modal Fusion**: Joint modeling of Diffusion + T1/T2 + MT.

---

## Next Development Steps (Agent Prompts)

1.  **Differentiable Histology Bridge**:
    *   Create `dmipy_jax/validation/histology.py`.
    *   Load histology images (e.g. from Zenodo).
    *   Differentiate fitting error w.r.t histology-predicted signals.

2.  **Multi-Modal Fusion**:
    *   Implement joint T1-Diffusion models (e.g. relaxation-diffusion spectrum).
