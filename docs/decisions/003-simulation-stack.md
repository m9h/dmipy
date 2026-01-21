# 3. Simulation Stack Strategy

Date: 2026-01-20

## Status

Accepted

## Context

We require a high-fidelity simulation engine to validate our analytical models. This engine must simulate:
1.  **Bloch Equations**: For MRI pulse sequence evolution (T1/T2 relaxation, RF pulses).
2.  **Stochastic Differential Equations (SDEs)**: For Monte Carlo diffusion particle simulations.

Standard Python solvers (like `scipy.integrate.odeint`) are not JIT-compatible and cannot be easily differentiated through to optimize acquisition parameters.

## Decision

We adopt `diffrax` as our numerical differential equation solver and formalize `JaxAcquisition` as a Pytree.

### 1. Diffrax for Simulation
We use `diffrax` because it is:
*   **JIT-compatible**: Solves ODEs/SDEs inside compiled XLA kernels.
*   **Differentiable**: We can compute gradients of the final signal with respect to acquisition parameters (like gradient strength or duration) using adjoint sensitivity methods.

**Workflows:**
*   **Bloch**: `diffrax.diffeqsolve` with `Tsit5` or `Dopri5` solvers.
*   **Diffusion**: `diffrax.diffeqsolve` with `EulerHeun` for SDEs.

### 2. JaxAcquisition Pytree
The acquisition definition (b-values, gradient directions, pulse timings) is no longer a static dictionary. It is an `eqx.Module` called `JaxAcquisition`.

**Why?**
By registering the acquisition as a Pytree, `diffrax` solvers can accept it as an input argument. This allows `jax.jit` to cache the simulation kernel once, and re-run it efficiently even if we change the b-values or gradient directions, provided the array shapes remain constant. This is critical for acquisition optimization algorithms.

## Consequences

*   **Positive**: Unlocks "differentiable physics" – we can optimize the MRI sequence to maximize sensitivity to a specific tissue parameter.
*   **Negative**: Diffrax has a learning curve compared to standard ODE solvers (handling `SaveAt`, `PIDController` stepsize logic).
