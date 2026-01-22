#!/bin/bash
set -e

SUB=sub-01
PREPROC_DIR=/home/mhough/datasets/ds001957-study/derivatives/preproc_qsiprep/$SUB
OUT_DIR=/home/mhough/datasets/ds001957-study/derivatives/super_resolution/$SUB/dwi
mkdir -p $OUT_DIR

DWI=$PREPROC_DIR/dwi/dwi.nii.gz
T1=$PREPROC_DIR/anat/t1.nii.gz

# 1. Regrid DWI to T1 space using Sinc Interpolation (MRtrix)
# We assume the images are roughly aligned handled by header (or we rely on previous registration)
# But standard "Upsampling" just changes voxel size.
# If we want Co-registered baseline, we should rigid reg first.

# Rigid Reg (MRtrix) using b0
mrconvert $DWI $OUT_DIR/b0.nii.gz -coord 3 0 -axes 0,1,2 -force
mrregister $OUT_DIR/b0.nii.gz $T1 -type rigid -rigid $OUT_DIR/rigid_mrtrix.txt -force

# Resample
# Transform DWI to T1 space
mrtransform $DWI -linear $OUT_DIR/rigid_mrtrix.txt -template $T1 -interp sinc $OUT_DIR/sub-01_desc-baseline_dwi.nii.gz -force

echo "Baseline Complete."
