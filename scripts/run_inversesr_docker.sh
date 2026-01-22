#!/bin/bash
set -e

# Configuration
IMAGE_NAME="inversesr:latest"
DOCKERFILE_DIR="/home/mhough/dev/dmipy/benchmarks/external/InverseSR"
WORK_DIR="/home/mhough/dev/dmipy"
SCRIPT_TO_RUN="${1:-scripts/batch_inversesr.py}"
ARGS="${@:2}"

echo ">>> Building Docker image..."
docker build -t "$IMAGE_NAME" "$DOCKERFILE_DIR"

echo ">>> Running in Docker..."
# We mount /home/mhough to /home/mhough to preserve all absolute paths 
# used in the scripts (e.g. DATASET_ROOT, INVERSESR_ROOT)
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v /home/mhough:/home/mhough \
    -w "$WORK_DIR" \
    "$IMAGE_NAME" \
    python3 "$SCRIPT_TO_RUN" $ARGS
