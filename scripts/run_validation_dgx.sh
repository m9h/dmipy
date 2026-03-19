#!/usr/bin/env bash
# ==============================================================================
# DGX Spark Validation Runner
#
# Runs the full validation suite for the Forward-Modeling & SBI extensions.
# Designed for NVIDIA DGX Spark with JAX CUDA 12 backend.
#
# Usage:
#   # On DGX directly:
#   bash scripts/run_validation_dgx.sh
#
#   # Via Docker:
#   docker build -t dmipy-validate .
#   docker run --gpus all -v $(pwd)/results:/app/results dmipy-validate \
#       bash scripts/run_validation_dgx.sh
#
#   # Via Singularity/Apptainer (HPC):
#   singularity exec --nv dmipy.sif bash scripts/run_validation_dgx.sh
# ==============================================================================

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/validation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/validation.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
}

# ==============================================================================
# 0. Environment check
# ==============================================================================
log "=========================================="
log "DGX Spark Validation Suite"
log "=========================================="

log "Python: $(python --version 2>&1)"
python -c "
import jax
print(f'JAX version:  {jax.__version__}')
print(f'Devices:      {jax.devices()}')
print(f'Default:      {jax.default_backend()}')
" 2>&1 | tee -a "$LOG"

# ==============================================================================
# 1. Unit tests (fast sanity check)
# ==============================================================================
log ""
log "--- Phase 0: Unit tests ---"
python -m pytest tests/test_pipeline_simulator.py tests/test_pipeline_checkpoint.py \
    tests/test_library_storage.py tests/test_metrics.py \
    tests/test_library_generator.py tests/test_matcher.py \
    tests/test_hybrid_init.py \
    --override-ini="addopts=" --noconftest -q 2>&1 | tee -a "$LOG"

log "Unit tests passed"

# ==============================================================================
# 2. Benchmarks
# ==============================================================================
log ""
log "--- Phase 1: Library generation benchmark ---"
python benchmarks/benchmark_library_gpu.py 2>&1 | tee -a "$LOG"

log ""
log "--- Phase 2: Dictionary matching benchmark ---"
python benchmarks/benchmark_matching.py 2>&1 | tee -a "$LOG"

# ==============================================================================
# 3. Validation experiments
# ==============================================================================
log ""
log "--- Phase 3: FORCE replication (crossing-angle detection) ---"
python validation/validate_force_replication.py 2>&1 | tee -a "$LOG"
cp -f validation/force_replication_results.npz "$RESULTS_DIR/" 2>/dev/null || true

log ""
log "--- Phase 4: Nottingham SBI replication (Ball+2Stick NPE) ---"
python validation/validate_nottingham_sbi.py 2>&1 | tee -a "$LOG"
cp -f validation/nottingham_sbi_results.npz "$RESULTS_DIR/" 2>/dev/null || true

log ""
log "--- Phase 5: Alicante SBI replication (DTI minimal acquisition) ---"
python validation/validate_alicante_sbi.py 2>&1 | tee -a "$LOG"
cp -f validation/alicante_sbi_results.npz "$RESULTS_DIR/" 2>/dev/null || true

log ""
log "--- Phase 6: Bingham-NODDI SBI (anisotropic dispersion) ---"
python validation/validate_bingham_sbi.py 2>&1 | tee -a "$LOG"
cp -f validation/bingham_sbi_results.npz "$RESULTS_DIR/" 2>/dev/null || true

log ""
log "--- Phase 7: End-to-end pipeline (synthetic volume) ---"
python validation/validate_e2e_pipeline.py 2>&1 | tee -a "$LOG"

# ==============================================================================
# 4. Summary
# ==============================================================================
log ""
log "=========================================="
log "Validation complete"
log "Results directory: $RESULTS_DIR"
log "=========================================="
ls -lh "$RESULTS_DIR/" | tee -a "$LOG"
