#!/bin/bash
# set -e

echo "=== Staging QSIPrep Data ==="
bash examples/sr_stage_qsiprep.sh

echo "=== Running Baseline (MRtrix) ==="
bash examples/sr_baseline.sh

echo "=== Running MMORF (FSL) ==="
bash examples/sr_mmorf.sh

echo "=== Running JTV (JAX/SCICO) ==="
uv run python examples/sr_jtv.py

echo "=== Running INR (JAX/Equinox) ==="
uv run python examples/sr_inr.py
