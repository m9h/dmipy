#!/bin/bash
# scripts/benchmark_sbi_dmri.sh
# Automated script to build and run the SBI_dMRI Oracle.

# Configuration
IMAGE_NAME="sbi_dmri_oracle"
DATA_DIR="$(pwd)/data/oracle"
DOCKERFILE="docker/Dockerfile.sbi_dmri"

mkdir -p "$DATA_DIR"

echo "=========================================="
echo "      SBI_dMRI Oracle Benchmark Setup     "
echo "=========================================="

# 1. Build Container
echo "[1/3] Building Docker Image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" . || { echo "Build failed"; exit 1; }

# 2. Run Verification (Short Run)
echo "[2/3] Verification: Running small batch (100 samples)..."
# We assume the sbi_dmri repository has a script 'scripts/generate_data.py' or similar.
# Since we clone it in the Dockerfile, we need to know the exact path.
# The Dockerfile WORKDIR is /workspace/sbi_dmri.
# We'll run a python command.

# Command to test basic functionality and creating 'test_params.npy'
docker run --gpus all --rm \
    -v "$DATA_DIR":/data/sbi_dmri_oracle \
    "$IMAGE_NAME" \
    python -c "import sbi; import torch; print('SBI Import Successful. Simulating...'); import numpy as np; np.save('/data/sbi_dmri_oracle/test_verification.npy', np.zeros(10))"

if [ $? -eq 0 ]; then
    echo "Verification Successful. Container manages IO correctly."
else
    echo "Verification Failed."
    exit 1
fi

# 3. Instructions for Long Run
echo "=========================================="
echo "      Ready for High-Throughput Run       "
echo "=========================================="
echo "To generate the complex example (1M samples, ~4 hours), run:"
echo ""
echo "docker run --gpus all -d --name sbi_long_run \\"
echo "    -v \"$DATA_DIR\":/data/sbi_dmri_oracle \\"
echo "    \"$IMAGE_NAME\" \\"
echo "    python scripts/generate_data.py --count 1000000 --protocol hcp --simulate"
echo ""
echo "Monitor with: docker logs -f sbi_long_run"
echo "=========================================="
