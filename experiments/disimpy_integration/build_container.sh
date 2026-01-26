#!/bin/bash
set -e

# Build the container
docker build -t disimpy-ngc:25.12 -f Dockerfile.disimpy .

echo "Build complete: disimpy-ngc:25.12"
