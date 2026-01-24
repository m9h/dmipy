# Level 1 Architecture Scan Report

## Executive Summary
This report details the findings of the "Technical Audit Team" (AuditBot, TestBot, BuildBot) regarding the compliance of `dmipy-jax` with the established Strict Technical Standards. Significant deviations were found in type hygiene, unit handling, optimization verification, and build configuration.

## 1. AuditBot Findings (The Architect)
**Focus:** Mathematical implementation and separation of concerns.

### Standard Violations
*   **The "Naked Float" Scan:**
    *   **Violation:** `dmipy_jax/acquisition.py`: `JaxAcquisition` defines `bvalues`, `gradient_directions`, `delta`, `Delta`, `echo_time`, and `total_readout_time` as raw `jnp.ndarray` or `float`.
    *   **Violation:** `dmipy_jax/inverse/solvers.py`: `MicrostructureOperator` and `AMICOSolver` accept raw arrays/floats for parameters and data, lacking `unxt.Quantity` enforcement.
*   **Equinox Purity:**
    *   **Violation:** `JaxAcquisition` is a standard Python `dataclass` registered via `jax.tree_util.register_pytree_node`, rather than inheriting from `eqx.Module`.
    *   **Violation:** `JaxAcquisition.__post_init__` performs mutable state updates (`self.bvalues = ...`), violating the immutability requirement of the Equinox/JAX functional paradigm.
*   **The Scico Boundary:**
    *   **Violation:** No "Unit Sandwich" implementation found. Units are not stripped before entering Scico solvers (e.g., in `AMICOSolver.fit` or `GlobalOptimizer.solve_tv`), nor are they re-attached at the output.
*   **Type Hygiene:**
    *   **Violation:** Public methods in `JaxAcquisition` use generic type hints like `Union[jnp.ndarray, float]` instead of strict `jaxtyping` with named axes (e.g., `Float[Array, "N_meas 3"]`).

### Refactoring Candidates
*   **Refactor `JaxAcquisition`:** Convert to `eqx.Module`. Replace raw arrays with `unxt.Quantity`. Remove `__post_init__` mutation in favor of validated initialization or class methods.
*   **Implement Unit Sandwich:** Create a decorator or wrapper layer for `MicrostructureOperator` and Solvers that automatically strips `unxt` units before computation and re-attaches them to the result.

## 2. TestBot Findings (The QA Engineer)
**Focus:** Test robustness, numerical validity, and JIT safety.

### Standard Violations
*   **Convergence Verification:**
    *   **Violation:** `dmipy_jax/tests/test_inverse.py`: `test_global_optimizer_tv` checks for output plausibility (non-NaN, variance check) but does *not* verify that the cost function decreases monotonically or reaches a specific tolerance relative to the initial cost.
*   **Unit Stripping:**
    *   **Violation:** Tests (e.g., `test_inverse.py`) use `MockAcquisition` with raw `jnp.array` inputs. There are no tests verifying that the API accepts `unxt` Quantities and handles them correctly (stripping/re-attaching).
*   **Adjoint Test:**
    *   **Violation:** No explicit Adjoint Test ($\langle Ax, y \rangle = \langle x, A^T y \rangle$) found for `MicrostructureOperator` or other custom Linear Operators.
*   **Differentiation & JIT Safety:**
    *   **Violation:** No usage of `jax.test_util.check_grads` to verify gradient correctness.
    *   **Violation:** Tests are not explicitly wrapped in `@jax.jit` or checked for tracer leakage.

### Refactoring Candidates
*   **Enhance Inverse Tests:** Update `test_global_optimizer_tv` to inspect the solver's optimization history (e.g., `solver.diagnostics`) and assert cost reduction.
*   **Add Unit Tests:** Create a new test suite specifically for "Unit Sandwich" compliance, passing `unxt` objects to all public APIs.
*   **Add Adjoint/Grad Tests:** Implement standard property tests for all Operators.

## 3. BuildBot Findings (The CI Engineer)
**Focus:** Build environment, vendoring, and configuration.

### Standard Violations
*   **Dependency Pinning:**
    *   **Violation:** `pyproject.toml` lists `equinox` and `scico` without version pins.
    *   **Violation:** `unxt` is completely missing from `dependencies`.
*   **Precision Config:**
    *   **Violation:** `.github/workflows/ci.yml` does not set `jax.config.update("jax_enable_x64", True)` or `JAX_ENABLE_X64=True`. This is critical for Scico stability.
*   **CPU-only JAX:**
    *   **Violation:** `pyproject.toml` requests `jax[cuda12]`, but CI runs on standard `ubuntu-latest` (CPU). This mismatch may lead to inefficient builds or masking of CPU-fallback issues.
*   **Vendor Isolation:**
    *   **Violation:** No configuration (e.g., in `pyproject.toml` or `ruff.toml`) found to explicitly exclude `dmipy_jax/external` (Jemris, Pulseq) from linting.

### Refactoring Candidates
*   **Update Dependencies:** Add `unxt` to `pyproject.toml`. Pin `equinox`, `scico`, and `unxt` to known stable versions.
*   **Fix CI Config:** Add `JAX_ENABLE_X64=True` to the CI environment variables. Consider a dedicated CPU-only test job that installs `jax[cpu]`.
*   **Configure Linting:** Add a `[tool.ruff.lint.per-file-ignores]` section to `pyproject.toml` for `dmipy_jax/external/*`.
