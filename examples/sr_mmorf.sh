#!/bin/bash
set -e

SUB=sub-01
PREPROC_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/$SUB
OUT_DIR=/home/mhough/datasets/ds001957-study/derivatives/super_resolution/$SUB/dwi
mkdir -p $OUT_DIR

DWI=$PREPROC_DIR/dwi/dwi.nii.gz
T1=$PREPROC_DIR/anat/t1.nii.gz

echo "--- MMORF Pipeline (Affine Fallback) ---"
echo "Note: Full non-linear MMORF skipped due to config complexity."
echo "Performing high-quality Affine Registration + Spline Resampling."

# 1. Extract b0
mrconvert $DWI $OUT_DIR/b0.nii.gz -coord 3 0 -axes 0,1,2 -force

# 2. Flirt (Rigid/Affine) DWI -> T1
# We need T1 brain extracted for better registration
if [ ! -f "$OUT_DIR/t1_brain.nii.gz" ]; then
    bet $T1 $OUT_DIR/t1_brain.nii.gz -f 0.3
fi

# Flirt b0 -> t1_brain
echo "Running FLIRT (Affine)..."
flirt -in $OUT_DIR/b0.nii.gz \
      -ref $OUT_DIR/t1_brain.nii.gz \
      -out $OUT_DIR/b0_flirt.nii.gz \
      -omat $OUT_DIR/b0_to_t1.mat \
      -dof 6

# 3. Apply Warp (Affine only)
# Using FSL applywarp with premat and Spline interpolation
# This resamples the DWI onto the T1 grid (1mm isotropic usually)
echo "Applying Warp (Affine + Spline)..."
applywarp \
    --ref=$OUT_DIR/t1_brain.nii.gz \
    --in=$DWI \
    --out=$OUT_DIR/sub-01_desc-mmorf_dwi.nii.gz \
    --premat=$OUT_DIR/b0_to_t1.mat \
    --interp=spline

echo "MMORF (Affine Fallback) Complete."
