# Vendored Benchmark: dmipy-bayesian

**Date Vendored**: 2026-01-20
**Source**: [https://github.com/PaddySlator/dmipy-bayesian](https://github.com/PaddySlator/dmipy-bayesian)
**Version**: Git HEAD at time of vendoring (frozen)

## Purpose
This directory contains a frozen copy of the `dmipy-bayesian` repository. It serves as a **numerical oracle** (baseline) for verifying the new `dmipy-jax` MCMC and SBI implementations.

## Usage
**DO NOT INSTALL THIS PACKAGE IN YOUR MAIN ENVIRONMENT.**

It relies on outdated dependencies (`scipy==1.4.1`, legacy `dmipy`).
To run comparisons, you should verify the output of this code in an isolated environment (e.g., a separate `conda` environment or Docker container) and save the results (posteriors, corner plots) to be compared against `dmipy-jax`.

## Maintenance
**Do not modify the code in this directory.**
It is intentionally kept in its original state to represent the "previous generation" of methods. Any improvements should be made in the modern `dmipy-jax` codebase.
