# Agent Prompts for Real-World Datasets Showcases

These prompts guide the implementation of demonstrations using challenging real-world datasets: **ds003216 (7T High-Res)** and **IXI (Massive Scale)**.

> [!IMPORTANT]
> Both datasets require manual data acquisition steps as they are not standard OpenNeuro CLI-friendly downloads.

---

## Prompt 1: 7T Voxel-wise Interaction (ds003216)

**Title:** High-Resolution Microstructure Interaction Demo
**Objective:** Demonstrate that `dmipy-jax` fits physically consistent models across spatially coupled voxels, exploiting the 800µm resolution of 7T data.

**Context:**
We want to use `ds003216` (Comprehensive ultrahigh resolution human phantom). The data is accessible via S3 but paths are non-standard.
Subject: `sub-our_subject`.

**Instructions:**
1.  **Acquire Data:**
    *   Manually Inspect `ds003216` on OpenNeuro or S3 keys.
    *   Download the Diffusion NIfTI, bvals, and bvecs for `sub-our_subject`.
    *   *Alternative:* Use any other valid 7T dataset (e.g., MGH-HCP if available).
2.  **Create Script:** `examples/program_7T_interaction.py`.
3.  **Define Model:**
    *   Use `C1Stick` + `Zeppelin`.
    *   Important: Enable `spatial_coupling=True` (or equivalent TV regularization) in the solver.
4.  **Demonstration:**
    *   Crop a small 20x20x20 region (due to high res).
    *   Perturb the data with synthetic noise (optional).
    *   Show that the coupled fit recovers sharper boundaries (e.g. cortex/white matter interface) than voxel-wise fit.

---

## Prompt 2: Massive-Scale Throughput (IXI Dataset)

**Title:** "The 10-Minute Cohort" Benchmark
**Objective:** Prove `dmipy-jax` can process hundreds of subjects in the time standard tools process one.

**Context:**
The IXI Dataset contains ~600 healthy subjects. It is hosted on `brain-development.org`.

**Instructions:**
1.  **Acquire Data:**
    *   Download 10-20 subjects from `https://brain-development.org/ixi-dataset/` (DTI images).
    *   Or mock the scale by replicating the EDDEN subject 100 times in memory.
2.  **Create Script:** `benchmarks/program_throughput_cohort.py`.
3.  **Implementation:**
    *   Load all subjects into a single batched JAX array `(N_subjects, N_voxels, N_meas)`.
    *   Use `jax.pmap` (multi-GPU) or `jax.vmap` (single GPU) to fit them all simultaneously.
    *   Model: Standard DTI (Tensor extraction).
4.  **Metric:**
    *   Report "Subjects per Minute" processing speed.
    *   Goal: > 50 subjects/minute on a consumer GPU.
