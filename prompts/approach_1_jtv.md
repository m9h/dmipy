# Prompt: Joint Total Variation (JTV) Super-Resolution

## Context
We have low-resolution DWI data ($y$) and high-resolution structural data ($z$, T1w/FLAIR). We assume they are co-registered.
We want to reconstruct a high-resolution DWI volume ($x$) such that $Ax \approx y$ (consistency) and gradients in $x$ align with gradients in $z$ (JTV).

## Mathematical Formulation
$$ \hat{x} = \text{argmin}_x \frac{1}{2} \| D S x - y \|_2^2 + \lambda \sum_i \sqrt{ (\nabla x)_i^2 + \beta (\nabla z)_i^2 } $$
Where:
- $D$: Downsampling operator.
- $S$: Blur/PSF operator (optional, simulate acquisition).
- $x$: High-res DWI (unknown).
- $y$: Low-res DWI (observed).
- $z$: High-res Structural (observed).
- $\nabla$: Spatial gradient operator.

## Implementation Details
- **Library**: `scico` (or `dmipy-jax` wrappers if available).
- **Input**:
    - DWI: `derivatives/qsiprep/sub-01/dwi/sub-01*_desc-preproc_dwi.nii.gz` (and .bval/.bvec).
    - Mask: `derivatives/qsiprep/sub-01/dwi/sub-01*_desc-brain_mask.nii.gz`
    - T1: `derivatives/qsiprep/sub-01/anat/sub-01*_desc-preproc_T1w.nii.gz`
- **Steps**:
    1. Load NIfTI files.
    2. Normalize intensities (DWI and T1 should be in roughly same magnitude range or regularizer scaled).
    3. Define LinearOperator $A$ (Identity if working in upsampled space with mask, or Downsampling if solving super-res explicitly). *Recommendation: Solve Deconvolution+Restoration on T1 grid.*
    4. Define Functional $R(x; z)$ using `scico.functional.JointTotalVariation`.
    5. Solve using ADMM or L-BFGS-B (via `scico.optimize`).

## Output
- `derivatives/super_resolution/sub-01/dwi/sub-01_desc-jtv_dwi.nii.gz`
