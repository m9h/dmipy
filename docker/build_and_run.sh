#!/bin/bash
set -e

# Define image name
IMAGE_NAME="mdt-oracle:latest"
DOCKER_DIR="$(dirname "$0")"

echo "Building Docker image: $IMAGE_NAME..."
# Build the docker image
# We explicitly point to the Dockerfile.mdt
docker build -t "$IMAGE_NAME" -f "$DOCKER_DIR/Dockerfile.mdt" "$DOCKER_DIR"

echo "Running verification..."
# Run the container
# --rm removes the container after it exits
docker run --rm "$IMAGE_NAME"
