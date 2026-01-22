# Prompt: MMORF Multi-Modal Registration/Fusion

## Context
Use FSL's MMORF (Multi-Modal Optimized Registration Framework) to drive a high-dimensional non-linear registration of the DWI to the T1, utilizing all available scalar maps (FA, MD vs T1/FLAIR edges). This is "re-sampling" as super-resolution.

## Workflow
1.  **Preprocessing**:
    - Denoise & Degibbs DWI (Standard).
    - Fit Tensor (DTI) to get FA/MD maps from DWI.
2.  **Initial Registration**:
    - `flirt` (Rigid/Affine) DWI -> T1.
3.  **Non-Linear Registration (MMORF)**:
    - Inputs:
        - Image A: T1w (Scalar).
        - Image B: DWI-derived FA (Scalar) or Mean DWI.
        - *Note*: MMORF is best for multimodal. Mapping FA to T1 is tricky without a pseud-T1, but MMORF handles local correlation.
    - Config: High warp resolution (1-2mm).
4.  **Resampling**:
    - Apply the estimated warp to the original DWI series to resample it onto the 1mm T1 grid.

## Implementation Details
- **Library**: FSL (`fsl_anat`, `dtifit`, `mmorf`, `applywarp`).
- **Data Source**:
    - DWI: `derivatives/qsiprep/sub-01/dwi/sub-01*_desc-preproc_dwi.nii.gz`
    - T1: `derivatives/qsiprep/sub-01/anat/sub-01*_desc-preproc_T1w.nii.gz`
- **Script**: Bash script.

## Output
- `derivatives/super_resolution/sub-01/dwi/sub-01_desc-mmorf_dwi.nii.gz`
