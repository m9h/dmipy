#!/bin/bash
set -e

# Configuration
SUBJECTS=("sub-01" "sub-03")
QSI_DIR=/home/mhough/datasets/ds001957-study/derivatives/qsiprep

for SUB in "${SUBJECTS[@]}"; do
    echo "Processing $SUB..."
    OUT_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/$SUB
    mkdir -p $OUT_DIR/dwi
    mkdir -p $OUT_DIR/anat

echo "Searching for QSIPrep data for $SUB in $QSI_DIR..."

# Find T1
T1_FILE=$(find $QSI_DIR/qsiprep/$SUB/anat -name "*_desc-preproc_T1w.nii.gz" | head -n 1)
T1_MASK_FILE=$(find $QSI_DIR/qsiprep/$SUB/anat -name "*_desc-brain_mask.nii.gz" | head -n 1)

ln -sf $T1_FILE $OUT_DIR/anat/t1.nii.gz
ln -sf $T1_MASK_FILE $OUT_DIR/anat/mask.nii.gz

# Find DWI
# Pattern: *desc-preproc_dwi.nii.gz
DWI_FILE=$(find $QSI_DIR -name "${SUB}*desc-preproc_dwi.nii.gz" | head -n 1)

# Find Mask
DWI_MASK=$(find $QSI_DIR -name "${SUB}*desc-brain_mask.nii.gz" | head -n 1)

if [ -z "$T1_FILE" ] || [ -z "$DWI_FILE" ]; then
    echo "Error: Could not find required QSIPrep files for $SUB."
    echo "Checked: $QSI_DIR"
    echo "Found T1: $T1_FILE"
    echo "Found DWI: $DWI_FILE"
    exit 1
fi

# Find bvals/bvecs (usually same base name as DWI)
# If DWI is file.nii.gz, bval is file.bval
BASE_NAME=${DWI_FILE%.nii.gz}
BVAL_FILE="$BASE_NAME.bval"
BVEC_FILE="$BASE_NAME.bvec"

if [ ! -f "$BVAL_FILE" ]; then
    echo "Warning: Bval file not found at $BVAL_FILE. Trying to find any bval..."
    BVAL_FILE=$(find $QSI_DIR -name "${SUB}*desc-preproc_dwi.bval" | head -n 1)
fi

if [ ! -f "$BVEC_FILE" ]; then
    echo "Warning: Bvec file not found at $BVEC_FILE. Trying to find any bvec..."
    BVEC_FILE=$(find $QSI_DIR -name "${SUB}*desc-preproc_dwi.bvec" | head -n 1)
fi

echo "Staging files to $OUT_DIR..."

# Symlink Data
ln -sf $T1_FILE $OUT_DIR/anat/t1.nii.gz
ln -sf $DWI_FILE $OUT_DIR/dwi/dwi.nii.gz
ln -sf $DWI_MASK $OUT_DIR/dwi/mask.nii.gz
cp $BVAL_FILE $OUT_DIR/dwi/dwi.bval
cp $BVEC_FILE $OUT_DIR/dwi/dwi.bvec

echo "Success! QSIPrep data staged for $SUB."
ls -l $OUT_DIR/dwi
ls -l $OUT_DIR/anat

done
