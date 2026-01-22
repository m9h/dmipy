# Task: Generate SBI_dMRI Oracle Data

**Objective**: Use the provided Docker/Singularity container to generate a "Gold Standard" test set from the `SBI_dMRI` repository. This dataset will serve as the ground truth oracle for benchmarking `dmipy-jax`.

**Infrastructure**: NVIDIA DGX / Spark Cluster.
**Container Definition**: `docker/Dockerfile.sbi_dmri`

## Instructions

1.  **Build the Container**:
    On a node with Docker access:
    ```bash
    docker build -t sbi_dmri_oracle -f docker/Dockerfile.sbi_dmri .
    ```
    *Note: If running on a cluster with Singularity/Apptainer:*
    ```bash
    singularity build sbi_dmri_oracle.sif docker-archive://sbi_dmri_oracle
    ```

2.  **Run Generation Scripts**:
    Run the generation for the standard benchmark protocols (e.g., HCP, ABCD).
    ```bash
    # Generic syntax
    docker run --gpus all -v /path/to/data_storage:/data/sbi_dmri_oracle sbi_dmri_oracle \
        python scripts/generate_data.py --protocol hcp --simulate --noise_type rician --count 10000
    ```

3.  **Output Validation**:
    Ensure the output contains:
    - `params.npy`: Ground truth microstructural parameters.
    - `signals.npy`: Simulated diffusion signals.
    - `protocol.json` or similar metadata.

4.  **Integration**:
    Once generated, point the `dmipy_jax.benchmarks.oracle_wrapper.SBIdMRIOracle` to the generated posterior files (if training) or use the data directly for "Inverse Validity" testing.

## Deliverables
- [ ] 10k Samples for HCP Protocol.
- [ ] 10k Samples for ABCD Protocol.
- [ ] Validated access via `SBIdMRIOracle`.
