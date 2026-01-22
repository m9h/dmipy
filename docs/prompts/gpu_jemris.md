# Task: GPU-Accelerated JEMRIS Simulation

**Objective**: Develop (or integrate) a GPU-accelerated version of the JEMRIS (Jülich Extensible MRI Simulator) framework. This is crucial for "Extensive MRI Simulations" where CPU-based Bloch simulations are too slow.

**Infrastructure**: NVIDIA DGX / Spark Cluster.
**Target**: `gpu_jemris` container or wrapper.

## Instructions

1.  **Survey Existing GPU Implementations**:
    - Investigate the [ISMRM GPU-JEMRIS](https://github.com/gpu-jemris) forks or similar.
    - If a stable CUDA version exists, prioritize containerizing it.

2.  **Containerize**:
    - Create `docker/Dockerfile.jemris_gpu`.
    - Base: `dune/jemris` (if Docker exists) or build from source on `nvidia/cuda`.
    - Dependencies: `CVODE`, `HDF5`, `CUDA Toolkit`.

3.  **Python Wrapper (Optional but Recommended)**:
    - Develop a `dmipy_jax.simulation.jemris_wrapper` that:
        - Writes the XML configuration sequence for JEMRIS.
        - Invokes the compiled binary.
        - Reads the output K-Space/Image.
    - *Goal*: `signal = jemris_simulate(sequence, phantom)` call signature.

4.  **Verification**:
    - Simulate a standard Diffusion Weighted EPI sequence on a numerical phantom (e.g. Shepp-Logan or BrainWeb).
    - Compare runtimes: CPU (standard JEMRIS) vs GPU. Target >50x speedup.

## Deliverables
- [ ] `docker/Dockerfile.jemris_gpu`
- [ ] Benchmark Report (Speedup factor).
- [ ] Integration with `dmipy-jax` via shell wrapper.
