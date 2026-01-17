# ADR 001: Adoption of Optimistix for Microstructure Model Fitting

## Status
Accepted

## Context
The project initially considered `optax` for all optimization tasks. However, microstructure modeling involves solving Non-Linear Least Squares (NLLS) problems, which map poorly to stochastic gradient descent methods optimized for neural network training. We needed a solver that offers:
1.  **Deterministic Convergence**: To ensure reproducibility.
2.  **High Precision**: Second-order convergence rates (like Levenberg-Marquardt).
3.  **Trust Region Methods**: To handle pathological curvature and NaNs in biophysical models.

## Decision
We have decided to split the optimization strategy:
-   **Microstructure Fitting**: Use **`optimistix`** (specifically `optimistix.least_squares` with Levenberg-Marquardt).
-   **Neural Network Training**: Use **`optax`** (when we eventually train bespoke estimators).

## Consequences
### Positive
-   **Speed**: Expect convergence in ~5-10 iterations vs 1000+ for Adam.
-   **Stability**: Trust Region methods prevent parameter excursions into invalid regions (e.g., negative diffusivity).
-   **Equivalency**: Matches the numerical rigor of C++ tools like CAMINO and AMICO.

### Negative
-   **Complexity**: Introduces a second optimization library dependency.
-   **Learning Curve**: Developers need to understand the distinction between the two engines.
