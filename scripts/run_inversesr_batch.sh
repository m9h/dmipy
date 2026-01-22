#!/bin/bash
# Wrapper script to run InverseSR batch processing with all dependencies

# Ensure weights are present (simple check)
if [ ! -d "benchmarks/external/InverseSR/decoder" ]; then
    echo "WARNING: 'decoder' directory not found in benchmarks/external/InverseSR."
    echo "Please download the pre-trained weights as per the InverseSR README."
fi

# Run with uv and required dependencies
uv run \
    --with mlflow \
    --with monai \
    --with tensorboard \
    --with matplotlib \
    --with scikit-image \
    --with scikit-learn \
    --with omegaconf \
    --with einops \
    --with torchvision \
    ${1:-scripts/batch_inversesr.py} "${@:2}"
