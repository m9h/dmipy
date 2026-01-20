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
uv pip install numpy scipy matplotlib h5py

# Try installing from PyPI
echo "Attempting install from PyPI..."
uv pip install pygpc || {
    echo "PyPI install failed. Attempting install from GitHub..."
    uv pip install git+https://github.com/dmueller43/pygpc.git
}

echo "Running Benchmark Script..."
python benchmarks/benchmark_pygpc_script.py


echo "Comparison Benchmark Complete."
