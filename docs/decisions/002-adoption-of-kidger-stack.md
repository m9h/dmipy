# 2. Adoption of Scientific JAX Stack (Kidger Stack)

Date: 2026-01-16

## Status

Accepted

## Context

We need a robust framework for building physical models that are fully compatible with JAX's JIT compilation and differentiation transformations. Standard Python classes and `scipy.optimize` do not naturally support these requirements. Specifically, we need:
-   **Neural network/Model state management:** Provided by `equinox`.
-   **Non-linear least squares solvers:** Provided by `optimistix` (replaces `scipy.optimize` for precision).
-   **Robust linear solvers:** Provided by `lineax`.
-   **Differential equation solvers:** Provided by `diffrax`.
-   **Type checking for array shapes:** Provided by `jaxtyping`.

## Decision

We are abandoning standard Python classes for physical models and `scipy.optimize` for fitting.

**The Standard:**
1.  All physical models **MUST** inherit from `eqx.Module`.
2.  All array inputs **MUST** be typed using `jaxtyping` (e.g., `Float[Array, "batch 3"]`).

## Consequences

### Positive
-   **Full JIT-compatibility:** Models and solvers can be JIT-compiled end-to-end.
-   **Shape Safety:** `jaxtyping` prevents shape errors at runtime and improves code readability.
-   **Scientific Precision:** `optimistix` allows using 2nd-order solvers (Levenberg-Marquardt) which are superior for scientific fitting compared to first-order optimizers often used in DL.
-   **Unified Stack:** Using the "Kidger Stack" provides a cohesive set of tools designed to work together.

### Negative
-   **Learning Curve:** Developers need to learn `equinox` patterns and `jaxtyping` syntax.
-   **Migration Effort:** Existing standard Python classes need to be refactored to `eqx.Module`.
