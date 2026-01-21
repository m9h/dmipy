# Agent Prompts for EDDEN Dataset Showcases

These prompts are designed for an Agentic AI (like me) to implement advanced demonstrations using the EDDEN dataset (`ds004910`).

## Prompt 1: Whole-Brain High-Throughput NODDI Mapping

**Title:** Implement Whole-Brain NODDI Benchmark
**Objective:** Demonstrate the extreme speed of `dmipy-jax` by fitting a standard NODDI model to a full whole-brain volume (or single slice) from the EDDEN dataset.

**Context:**
We have the EDDEN dataset downloaded at `benchmarks/data/edden`. We want to show that JAX can fit complex models to massive numbers of voxels (e.g. 100k+) in seconds, a task that typically takes minutes or hours.

**Instructions:**
1.  **Create Script:** `examples/program_high_throughput_noddi.py`.
2.  **Load Data:** Use `nibabel` to load `sub-01/ses-02/dwi`. Load the whole volume (or a complete middle slice, e.g., z=50).
3.  **Define Model:** Implement the standard Watson-NODDI model using `dmipy_jax.signal_models`:
    *   **Intra-axonal:** `C1Stick` (parameter: `mu`, fixed `lambda_par=1.7e-9`).
    *   **Extra-axonal:** `Zeppelin` (parameter: `lambda_par`, `lambda_perp` linked via tortuosity constraint, shared `mu` with stick).
    *   **CSF:** `Ball` (fixed `lambda_iso=3.0e-9`).
    *   *Note: If tortuosity is hard to implement quickly, simpler 2-compartment Stick+Ball or Stick+Zeppelin is acceptable.*
4.  **Flatten Data:** Reshape the volume to `(N_voxels, N_meas)`.
5.  **Benchmark Fit:**
    *   Use `JaxMultiCompartmentModel.fit(..., method="LBFGSB")` (or `Optimistix` if stable).
    *   Time the `fit()` call (excluding compilation time if possible, or report both warm/cold).
6.  **Report:** Print the "Voxels per Second" throughput. It should be >100,000 voxels/sec on CSV/GPU.

---

## Prompt 2: Protocol Optimization (OED) Validation

**Title:** Validate OED Subsampling on Real Data
**Objective:** Prove that Optimal Experimental Design (OED) works by subsampling the EDDEN dataset.

**Context:**
EDDEN has a dense acquisition (~40+ directions). We want to show that selecting the "Right" 20 directions yields better results than a "Random" 20 directions.

**Instructions:**
1.  **Create Script:** `examples/program_oed_validation.py`.
2.  **Load Data:** Load EDDEN `sub-01/ses-02/dwi`.
3.  **Define Task:** We want to estimate Axon Estimator (Stick parameters).
4.  **Run OED:**
    *   Use `dmipy_jax.optimization.oed` (or implement a simple Condition Number minimization wrapper).
    *   Select the "Best 15" gradients from the full bvecs list that minimize the Cramer-Rao Bound (CRB) for a standard Stick+Ball model.
    *   Also select a "Random 15" subset.
5.  **Validation:**
    *   **Ground Truth:** Fit the model to the **FULL** dataset -> `Params_GT`.
    *   **OED Performance:** Fit to "Best 15" -> `Params_OED`. Compute Error vs GT.
    *   **Random Performance:** Fit to "Random 15" -> `Params_Rand`. Compute Error vs GT.
6.  **Success Metric:** Show that `Error(OED) < Error(Random)`. This validates our OED tooling on real data.
