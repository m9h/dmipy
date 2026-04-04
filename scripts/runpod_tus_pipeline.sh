#!/usr/bin/env bash
# RunPod: TUS SCI Head Pipeline
#
# Setup on RunPod:
#   1. Launch a pod with PyTorch 2.x template (has CUDA)
#   2. SSH in or use web terminal
#   3. Run this script:
#      curl -sSL https://raw.githubusercontent.com/m9h/sbi4dwi/master/scripts/runpod_tus_pipeline.sh | bash
#
# Or clone and run:
#   git clone https://github.com/m9h/sbi4dwi.git
#   cd sbi4dwi
#   bash scripts/runpod_tus_pipeline.sh
set -euo pipefail

echo "=== RunPod TUS Pipeline Setup ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone repo if not already present
if [ ! -d "sbi4dwi" ] && [ ! -f "pyproject.toml" ]; then
    echo "Cloning sbi4dwi..."
    git clone https://github.com/m9h/sbi4dwi.git
    cd sbi4dwi
elif [ -f "pyproject.toml" ]; then
    echo "Already in sbi4dwi repo"
    git pull origin master 2>/dev/null || true
else
    cd sbi4dwi
    git pull origin master 2>/dev/null || true
fi

# Setup venv
echo "Setting up environment..."
uv venv .venv --python 3.11 2>/dev/null || uv venv .venv
source .venv/bin/activate
uv pip install "jax[cuda12]" jwave optax equinox scipy "numpy<2" xarray h5py
uv pip install -e .

# Check for SCI head data
SCI_DATA="data/SCI_headmodel/extracted/HeadMesh.mat"
if [ ! -f "$SCI_DATA" ]; then
    echo ""
    echo "SCI head data not found at $SCI_DATA"
    echo "Upload it to the pod:"
    echo "  mkdir -p data/SCI_headmodel/extracted/"
    echo "  # Upload HeadMesh.mat via RunPod file manager or scp"
    echo ""
    echo "Then re-run: python scripts/dgx_tus_sci_head.py"
    exit 1
fi

# Run pipeline
echo ""
echo "=== Running TUS Pipeline ==="
python scripts/dgx_tus_sci_head.py

echo ""
echo "=== Complete ==="
