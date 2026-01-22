#!/bin/bash
set -e

# Data Paths
SUB=sub-01
RAW_DWI=/home/mhough/datasets/ds001957-study/ds001957/$SUB/dwi/${SUB}_acq-epidtidir64AP_dwi.nii.gz
RAW_BVAL=/home/mhough/datasets/ds001957-study/ds001957/$SUB/dwi/${SUB}_acq-epidtidir64AP_dwi.bval
RAW_BVEC=/home/mhough/datasets/ds001957-study/ds001957/$SUB/dwi/${SUB}_acq-epidtidir64AP_dwi.bvec
RAW_T1=/home/mhough/datasets/ds001957-study/ds001957/$SUB/anat/${SUB}_acq-t1grefsp3dsagiso1mm_echo-1_T1w.nii.gz
RAW_FLAIR=/home/mhough/datasets/ds001957-study/ds001957/$SUB/anat/${SUB}_acq-t2mx3dflairsagiso1mm_FLAIR.nii.gz

OUT_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc/$SUB/dwi
mkdir -p $OUT_DIR

echo "Processing Subject: $SUB"

# 1. Denoise
if [ ! -f "$OUT_DIR/dwi_den.nii.gz" ]; then
    echo "Denoising..."
    dwidenoise $RAW_DWI $OUT_DIR/dwi_den.nii.gz
fi

# 2. Degibbs
if [ ! -f "$OUT_DIR/dwi_den_deg.nii.gz" ]; then
    echo "Degibbs..."
    mrdegibbs $OUT_DIR/dwi_den.nii.gz $OUT_DIR/dwi_den_deg.nii.gz
fi

# Copy bvals first
cp $RAW_BVAL $OUT_DIR/dwi.bval
cp $RAW_BVEC $OUT_DIR/dwi.bvec

# 3. Brain Masking (Simple)
# Using dwi2mask on b0 with explicit grads
if [ ! -f "$OUT_DIR/mask.nii.gz" ]; then
    echo "Masking..."
    dwi2mask $OUT_DIR/dwi_den_deg.nii.gz $OUT_DIR/mask.nii.gz -fslgrad $OUT_DIR/dwi.bvec $OUT_DIR/dwi.bval
fi

# 4. Bias Correction (Optional)
# Linking preproc
ln -sf $OUT_DIR/dwi_den_deg.nii.gz $OUT_DIR/dwi_preproc.nii.gz

# 5. T1 Preparation (Copy to derivatives)
ANAT_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc/$SUB/anat
mkdir -p $ANAT_DIR
cp $RAW_T1 $ANAT_DIR/t1.nii.gz
cp $RAW_FLAIR $ANAT_DIR/flair.nii.gz

echo "Preprocessing Complete. Output: $OUT_DIR/dwi_preproc.nii.gz"
