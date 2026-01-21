#!/bin/bash
# Download EDDEN dataset (ds004910) sample for validation

# Exit on error
set -e

# Create data directory
mkdir -p data
cd data

# Install dataset if not exists
if [ ! -d "ds004910" ]; then
    echo "Installing ds004910..."
    datalad install https://github.com/OpenNeuroDatasets/ds004910.git
fi

cd ds004910

# Get specific session data
echo "Fetching sub-01/ses-02 diffusion data..."
datalad get sub-01/ses-02/dwi/

echo "Download complete."
