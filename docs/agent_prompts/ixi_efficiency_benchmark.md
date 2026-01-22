# Agent Prompt: IXI Dataset - Efficiency & Accuracy Benchmark

**Title:** The "10-Minute Cohort" Challenge (Efficiency, Accuracy, & Advanced Inference)

**Objective:**
Utilize the **IXI Dataset** (approx. 600 healthy subjects) to demonstrate the massive scalability and numerical robustness of `dmipy-jax`. The goal is to prove that we can fit an entire population study in minutes while maintaining higher physical accuracy than standard tools, and showcase advanced "Next-Gen" capabilities (Deep Learning & Uncertainty).

**Dataset Info:**
-   **Source:** [IXI Dataset](https://brain-development.org/ixi-dataset/)
-   **License:** Creative Commons CC BY-SA 3.0.
-   **Characteristics:** Approx 600 subjects. DTI acquisition (15 directions, b=1000).
-   **Challenge:** The large N requires extreme throughput. The low angular resolution (15 dirs) makes higher-order models unstable, requiring robust bounded solvers.

---

## Instructions for the Agent

1.  **Data Acquisition Strategy:**
    *   Since IXI is not on OpenNeuro, you cannot use `openneuro-py`.
    *   **Action:** Direct the user (or yourself via `curl`) to download a subset of 10-20 subjects for the prototype.
    *   *URL Pattern:* `http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar.gz` (Check valid URLs).
    *   *Alternative:* If download is too heavy, simulate the cohort by replicating a single subject (e.g., from EDDEN or a synthetic template) 600 times to match the array size `(600, 128, 128, 60, 15)`.

2.  **Benchmark Script:** `benchmarks/program_ixi_efficiency.py`
    *   **Load:** Load the data into a single JAX array $(N_{subj} \times N_{vox}, N_{meas})$.
    *   **Models:**
        *   **Standard DTI:** The baseline.
        *   **Robust Kurtosis:** If feasible with 15 dirs + b=1000 (likely ill-posed, good for testing reliability).
        *   **Biophysical:** `C1Stick + Ball` (Simple 2-compartment).

3.  **Efficiency Demonstration (Speed):**
    *   Compare `dmipy-jax` (compiled) vs `standard loop` (simulated by disabling JIT or simple Loop).
    *   **Metric:** "Voxels per Second" and "Estimated Time for 600 Subjects".
    *   **Target:** fitting 600 subjects should be estimated to take < 10 minutes on a GPU.

4.  **Accuracy Demonstration (Quality):**
    *   Compare **Unconstrained LM** (`Optimistix` default) vs **Bounded LBFGS** (`VoxelFitter`).
    *   **Metric 1: Physical Validity.** Count the % of voxels with negative diffusivities or NaNs.
    *   **Metric 2: Residual Error.** Compute RMSE of the fit.
    *   **Hypothesis:** `VoxelFitter` will be slightly slower but will have 0% invalid physics, whereas Unconstrained LM will fail on the noisy, low-direction data.

5.  **Amortized Inference Demonstration (Deep Learning):**
    *   **Goal:** Train a Neural Network to predict parameters instantaneously.
    *   **Method:**
        *   Train a small MLP (`eqx.nn.MLP`) on the first 50 subjects ("Training Set").
        *   Predict the remaining 550 subjects ("Test Set") with `vmap(net)`.
    *   **Metric:** Compare Inference Time vs Conventional fit. Expect milliseconds vs minutes.

6.  **Uncertainty Quantification (Bayesian VI):**
    *   **Goal:** Map confidence on this low-quality (15-direction) data.
    *   **Method:**
        *   Use `VIMinimizer` (Variational Inference) on a representative slice of 1 subject.
        *   Output: Mean + Standard Deviation maps.
    *   **Report:** Highlight regions where StdDev is high (uncertainty) to show clinical safety value.

7.  **Deliverables:**
    *   A single unified report `benchmarks/reports/ixi_comprehensive_benchmark.md` summarizing:
        | Method | Solver | Speed (vox/s) | Neg Diffusivity % | RMSE |
        |--------|--------|---------------|-------------------|------|
        | DTI | WLS (Exact) | ... | ... | ... |
        | Stick+Ball | Unconstrained LM | ... | ... | ... |
        | Stick+Ball | Bounded LBFGS | ... | ... | ... |
        | Stick+Ball | Amortized Net | ... | N/A | ... |

    *   *Visuals:* Slices showing Parameter Maps (Conventional) vs Uncertainty Maps (VI).
