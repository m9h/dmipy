#!/usr/bin/env bash
# DGX Spark: Full TUS pipeline test
# Run from sbi4dwi root: bash scripts/dgx_tus_pipeline.sh
set -euo pipefail

echo "=== DGX TUS Pipeline Test ==="
echo "Date: $(date)"
echo ""

# 0. Check GPU
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 1. Pull latest
echo "--- Pull latest ---"
git pull origin master
echo ""

# 2. Sync deps
echo "--- Sync dependencies ---"
uv sync
uv pip install jwave optax
echo ""

# 3. Check SCI head data
SCI_DATA="data/SCI_headmodel/extracted/HeadMesh.mat"
if [ ! -f "$SCI_DATA" ]; then
    echo "ERROR: SCI head data not found at $SCI_DATA"
    echo "Copy it from Google Drive:"
    echo "  mkdir -p data/SCI_headmodel/extracted/"
    echo "  cp /path/to/HeadMesh.mat data/SCI_headmodel/extracted/"
    exit 1
fi
echo "SCI head data found: $SCI_DATA"
echo ""

# 4. Tier 1: Fast CPU unit tests
echo "--- Tier 1: Unit Tests ---"
uv run pytest dmipy_jax/tests/biophysics/test_acoustic.py \
              dmipy_jax/tests/biophysics/test_mesh_rasterizer.py \
              --noconftest -v --tb=short
echo ""

# 5. Tier 2: j-Wave adapter tests (includes simulation)
echo "--- Tier 2: j-Wave Adapter Tests ---"
uv run pytest dmipy_jax/tests/biophysics/test_jwave_adapter.py \
              --noconftest -v --tb=short -m "not slow" 2>/dev/null || \
uv run pytest dmipy_jax/tests/biophysics/test_jwave_adapter.py \
              --noconftest -v --tb=short
echo ""

# 6. SCI Head Full Pipeline
echo "--- Tier 3: SCI Head Model Pipeline ---"
uv run python scripts/dgx_tus_sci_head.py
echo ""

echo "=== All DGX Tests Complete ==="
