# Research Brief: Advanced Synergies in Differentiable MRI

## 1. Rough Paths and Neural CDEs: The Universal Waveform Model

**The Question**: "Do rough paths have anything to offer microstructure MRI?"
**The Answer**: **Yes.** They offer a unifying algebra for "General Waveform" imaging.

### The Problem
Currently, we treat Stejskal-Tanner (PGSE) and Oscillating Gradients (OGSE) as different "pulse sequences" with different analytical formulas. This handles smooth paths poorly.

### The Opportunity: Neural CDEs
A **Neural Controlled Differential Equation (CDE)** solves:
$$ d\mathbf{z}(t) = f_\theta(\mathbf{z}(t)) \, dX(t) $$
where $X(t)$ is a driving path. This is the continuous analogue of an RNN.

**Proposal**: Treat the **Gradient Waveform** $\mathbf{G}(t)$ (or the effective q-trajectory $\mathbf{q}(t)$) as the driving path $X(t)$.
*   **Input**: The "Signature" of the gradient path (iterated integrals of $G(t)$).
    *   Rough Path theory tells us the Signature uniquely characterizes the path's effect on nonlinear systems.
*   **Model**: A Neural CDE that learns the tissue response dynamics.
*   **Output**: The magnetization evolution $M(t)$.

**Why this is huge**:
*   It creates a **Waveform-Agnostic Model**. You train it on random squiggly gradients, and it generalizes to PGSE, OGSE, or any future pulse sequence.
*   It replaces the "Analytical Formula" lookup table with a single dynamical system learner.

## 2. JAX-MD + ICNN: The "Emulator" Synergy

**The Question**: "What additional synergy allows us to bring new physics to bear?"
**The Answer**: **The Simulator trains the Theory.**

We have two powerful engines in `dmipy-jax` that are currently disconnected:
1.  **JAX-MD Particle Engine** (`core/particle_engine.py`): Ground-truth physics, slow, accurate.
2.  **Neural Signal / ICNN** (`biophysics/neural_signal.py`): Interpretable approximation, fast, differentiable.

**Proposal: The Emulator Loop**
1.  **Generate**: Run JAX-MD 10,000 times on complex, irregular geometries (e.g., from generated phantom meshes).
2.  **Train**: Use this massive synthetic dataset to train the `NeuralSignalModel` (ICNN).
3.  **Result**: The ICNN becomes a **Differentiable Surrogate (Emulator)** of the particle simulation.
    *   We can now invert the particle physics (which is non-differentiable or chaotic) by inverting the smooth ICNN surrogate.
    *   We "distill" the complex Monte Carlo physics into a convex neural network.

## 3. Physics Expansion: T2-Diffusion-Exchange

**The Question**: "What additional MRI physics?"
**The Answer**: **Relaxation and Permeability.**

### Bloch-Torrey-SDE
We currently simulate pure diffusion. We should add **T2 Relaxation**.
*   **Particle Engine**: Each particle carries a "magnetization" weight $w_i(t)$ that decays: $dw_i = - (1/T_2(\mathbf{r})) w_i dt$.
*   **Exchange**: If particles cross compartment boundaries (membrane permeability), their effective $T_2$ changes.
*   **Synergy**: This enables **Joint T2-Diffusion Spectroscopy**.
    *   We can differentiate the "spectra" (peaks in the T2-D plane) w.r.t. tissue properties.
    *   *Algebraic Link*: The spectrum is a sum of exponentials. Our **Algebraic Initializers** (which we just built for bi-exponentials) are effectively **Spectral Peak Finders**. We can use them to initialize the spectral fit.
