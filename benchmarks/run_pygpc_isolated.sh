#!/bin/bash
set -e

# Define environment name
VENV_DIR=".venv_pygpc"

echo "Creating isolated environment for PyGPC..."

# Create venv using uv
uv venv $VENV_DIR --seed

# Activate
source $VENV_DIR/bin/activate

echo "Installing PyGPC and dependencies..."
# Install numpy and scipy first, then pygpc
# PyGPC might not be on PyPI or might have specific requirements. 
# Assuming `pip install pygpc` works, or we might need git.
# Checking PyGPC installation instructions: usually `pip install pygpc`
uv pip install numpy scipy matplotlib h5py
# PyGPC requires specific installation, let's try standard pip first.
uv pip install pygpc

echo "Running Benchmark Script..."
python benchmarks/benchmark_pygpc_script.py

echo "Comparison Benchmark Complete."
