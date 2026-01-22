#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t sbi_oracle .

# Run the container with GPU support
# Mounting the local 'data' directory to '/data' in the container
echo "Running SBI Oracle Simulation..."
# Explicitly using --gpus all for DGX/Spark support
docker run --gpus all --rm -v $(pwd)/data:/data sbi_oracle python generate_connectome_oracle.py
