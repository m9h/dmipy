
# Prompt: Algebraic Initializers via Grobner Bases

## Context
We are entering **Phase 4.1** of the Roadmap ("The Algebraic Era"). We have powerful diffrax-based solvers (SDEs, ODEs) and Normalizing Flows, but they are computationally expensive to initialize from scratch.
Standard fitting often gets stuck in local minima.
We want to use **Algebraic Geometry** to find "Instant" analytical initializers for our compartment models.

## Objective
Create the module `dmipy_jax/fitting/algebraic.py`.
This module should use `sympy` to compute **Grobner Bases** of the polynomial equations describing simple signal models (e.g., Cylinder, Stick, Zeppelin).
The goal is to find a **rational function approximation** $\theta \approx P(S) / Q(S)$ that maps signal invariants directly to model parameters.

## Tasks
1.  **Implement `SymbolicInverter` class**:
    *   Input: A `dmipy_jax` signal model (e.g. `Cylinder`).
    *   Method: `algebraize()`: Convert the model equations into polynomial form (e.g. replace $\exp(-bx)$ with variables).
    *   Method: `compute_grobner()`: Use `sympy.groebner` to solve for parameters in terms of signal moments.
    *   Method: `lambdify_initializer()`: Compile the resulting solution into a JAX-compatible function.

2.  **Specific Example: The "Linearized" Tensor**:
    *   Implement the standard Log-Linear Tensor fit $(D = - \frac{1}{b} \ln(S/S_0))$ as a trivial algebraic initializer example.

3.  **Advanced Example: The "Stick" Model**:
    *   For a Stick model $S = f \exp(-b D_{par} (\mathbf{g} \cdot \mathbf{n})^2)$, find an algebraic estimator for $f$ and $D_{par}$ using 2 shells.

## Deliverables
-   `dmipy_jax/fitting/algebraic.py`
-   `experiments/test_algebraic_init.py` (Verify it's < 1ms and distinct from local minima)
