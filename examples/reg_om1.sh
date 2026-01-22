#!/bin/bash
set -e

# Configuration
# Accepting Subject ID as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <sub-id>"
    exit 1
fi

SUB=$1
PREPROC_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/$SUB
OUT_DIR=/home/mhough/datasets/ds001957-study/derivatives/om1_registration/$SUB
mkdir -p $OUT_DIR

# Template
TEMPLATE_T1=$FSLDIR/data/omm/Oxford-MM-1/OMM-1_T1_brain.nii.gz

# Input
T1=$PREPROC_DIR/anat/t1.nii.gz
DWI=$PREPROC_DIR/dwi/dwi.nii.gz

echo "--- Registering $SUB to OM-1 Template ---"

# 1. Register T1 -> OM-1 T1 (Affine)
# Using FLIRT
echo "Running FLIRT (T1 -> OM-1)..."
flirt -in $T1 \
      -ref $TEMPLATE_T1 \
      -out $OUT_DIR/${SUB}_space-OM1_T1w.nii.gz \
      -omat $OUT_DIR/${SUB}_to_OM1.mat \
      -dof 12

# 2. Apply Transform to DWI
# DWI is assumed to be in AC-PC space aligned with T1 (from QSIPrep)
# Resampling to OM-1 Grid (1mm isotropic) with Spline Interpolation
echo "Resampling DWI to OM-1 Space..."
applywarp \
    --ref=$TEMPLATE_T1 \
    --in=$DWI \
    --out=$OUT_DIR/${SUB}_space-OM1_dwi.nii.gz \
    --premat=$OUT_DIR/${SUB}_to_OM1.mat \
    --interp=spline

echo "Registration Complete for $SUB."
echo "Output: $OUT_DIR/${SUB}_space-OM1_dwi.nii.gz"
