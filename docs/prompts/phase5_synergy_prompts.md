# Phase 5 Synergy Prompts

This document contains agent-ready prompts for the next advanced validation phases.

---

## 1. Neural CDEs for General Waveform Imaging

**Goal**: Implement a "Waveform-Agnostic" signal model using Neural Controlled Differential Equations (CDEs) driven by the Gradient Path Signature.

### Agent Prompt
```markdown
## Objective
Implement a Neural Controlled Differential Equation (CDE) model to learn the diffusion signal generic gradient waveforms $\mathbf{G}(t)$.

## Context
We currently use analytical formulas for PGSE/OGSE. We want a universal model $S(\mathbf{G}(t))$ that works for *any* gradient trajectory.
We will use `diffrax` for the CDE solver ($d\mathbf{z} = f(\mathbf{z}) dX$) and JAX for optimization.

## Requirements
1.  **Create `dmipy_jax/biophysics/neural_cde.py`**:
    *   Define `NeuralCDE(eqx.Module)`.
    *   Use `diffrax.ControlTerm` with `diffrax.LinearInterpolation` of the gradient waveform $\mathbf{G}(t)$.
    *   The hidden state $\mathbf{z}(t)$ evolves via an MLP vector field $f_\theta$.
    *   Map final state $\mathbf{z}(T)$ to Signal $S$.
2.  **Training Data**:
    *   Simulate signals for random oscillating gradients using `GaussianPhaseApproximation` (ground truth).
3.  **Training**:
    *   Train the Neural CDE to regress these signals.

## Constraints
*   Use `diffrax.diffeqsolve`.
*   Batch over waveforms using `jax.vmap`.
```

### Experimental Plan (V&V)
*   **Verification (Code Correctness)**:
    *   Test the CDE on a constant gradient (Stejskal-Tanner). Verify it learns the mono-exponential decay $e^{-bD}$.
*   **Validation (Physics Generalization)**:
    *   Train on **Sine** waves (OGSE).
    *   Test on **Trapezoidal** pulses (PGSE).
    *   **Metric**: Mean Squared Error on the *unseen* waveform type.
    *   **Success**: Generalization Error < 1%.

---

## 2. The Emulator Loop (JAX-MD $\to$ ICNN)

**Goal**: Train a fast, differentiable ICNN surrogate on massive ground-truth data from the JAX-MD particle simulator.

### Agent Prompt
```markdown
## Objective
Create a "Digital Twin" of the JAX-MD Particle Simulator using an Input Convex Neural Network (ICNN).

## Context
Running Monte Carlo simulations is too slow for inverse fitting. We need a differentiable emulator.

## Requirements
1.  **Data Generation (`benchmarks/generate_emulator_data.py`)**:
    *   Use `dmipy_jax/core/particle_engine.py`.
    *   Simulate restricted diffusion in **Cylinders** of varying radii $R \in [1, 10]\mu m$.
    *   Generate 1,000 parameter combinations (Dataset size: 10k signals).
    *   Save dataset.
2.  **Emulator Training (`experiments/train_emulator.py`)**:
    *   Load `dmipy_jax/biophysics/neural_signal.py` (ICNN).
    *   Train `NeuralSignalModel` to regress the JAX-MD outputs.
    *   Loss: MSE + Convexity Regularization is inherent.
3.  **Emulator Verification**:
    *   Compare gradients $\nabla_q S_{ICNN}$ vs $\nabla_q S_{MC}$ (finite diff).

## Constraints
*   Use `optax` for training.
*   Ensure data generation uses GPU batching if possible.
```

### Experimental Plan (V&V)
*   **Verification**:
    *   Train Loss convergence.
    *   Check ICNN convexity constraints ($w \ge 0$).
*   **Validation**:
    *   **Inverse Crime**: Can we recover the cylinder radius $R$ from a synthetic signal by optimizing the *Emulator*?
    *   **Procedure**:
        1.  Measure Signal $S_{true}$ (JAX-MD).
        2.  Find $\hat{q} = \text{argmin} (S_{ICNN}(q) - S_{true})^2$.
        3.  Check error $|q_{true} - \hat{q}|$.
    *   **Success**: Parameter recovery within 5%.

---

## 3. T2-Diffusion Algebra (Spectral Initialization)

**Goal**: Extend algebraic initializers to solve for joint T2-Diffusion spectra (Bi-Exponential decay in 2D).

### Agent Prompt
```markdown
## Objective
Implement an Algebraic Initializer for Joint T2-Diffusion estimation.

## Context
We want to estimate compartments with distinct $(D, T_2)$ properties.
Signal Equation: $S(b, \tau) = \sum f_i \exp(-b D_i - \tau/T_{2,i})$.
This is a 2D sum-of-exponentials.

## Requirements
1.  **Extend `dmipy_jax/algebra/initializers.py`**:
    *   Implement `get_t2_diffusion_initializer(protocol)`.
    *   Select data points at differing $b$ (Diffusion weighting) and $\tau$ (Echo Time).
    *   Formulate the symbolic system $y_{jk} = \sum f_i X_i^j Y_i^k$.
    *   Solve using `SymbolicSolver` (or log-linear simplification).
2.  **Output**:
    *   Estimates for $(D_1, T_{2,1})$ and $(D_2, T_{2,2})$.

## Constraints
*   Assume 2 compartments.
*   Require sufficient $(b, \tau)$ sampling in protocol.
```

### Experimental Plan (V&V)
*   **Verification**:
    *   Synthetic test: $D=[1, 2], T_2=[50, 100]$.
    *   Protocol: Cross-product of 2 b-values and 2 echo times.
    *   Check exact recovery of parameters.
*   **Validation**:
    *   Initialize a full `LevenbergMarquardt` fit with these algebraic guesses.
    *   Compare convergence rate vs Random Initialization.
    *   **Success**: >10x speedup or avoidance of local minima.
```
