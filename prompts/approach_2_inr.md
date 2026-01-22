# Prompt: Implicit Neural Representation (INR) Super-Resolution

## Context
Instead of solving a convex optimization problem, we train a neural network to represent the DWI signal as a continuous function of coordinates, conditioned on the high-res structural data.

## Model Formulation
$$ S(r, b, g) = \text{MLP}( \gamma(r), z(r), b, g ) $$
Where:
- $r$: Spatial coordinates (x, y, z).
- $\gamma(r)$: Positional encoding (Fourier features).
- $z(r)$: Feature vector from T1w/FLAIR at coordinate $r$.
- $b, g$: B-value and Gradient vector.
- $S$: DWI signal.

## Implementation Details
- **Library**: `equinox`, `jax`.
- **Data Source**:
    - DWI: `derivatives/qsiprep/sub-01/dwi/sub-01*_desc-preproc_dwi.nii.gz`
    - Mask: `derivatives/qsiprep/sub-01/dwi/sub-01*_desc-brain_mask.nii.gz`
    - T1: `derivatives/qsiprep/sub-01/anat/sub-01*_desc-preproc_T1w.nii.gz`
- **Architecture**:
    - Inputs: Fourier Features of (x,y,z) + T1 intensity + FLAIR intensity.
    - Hidden: 4-8 layers of SIREN (Sine activation) or ReLU MLP.
    - Output: DWI signal intensity.
- **Training**:
    - Sample points $r_i$ from the low-res grid.
    - Loss: MSE between predicted $S(r_i)$ and observed voxel value $y_i$.
    - *Crucial*: We train *only* on the observed low-res voxels. The network generalizes to the continuum.
- **Inference (Super-Resolution)**:
    - Query the trained MLP at the high-res coordinates of the T1 grid.

## Output
- `derivatives/super_resolution/sub-01/dwi/sub-01_desc-inr_dwi.nii.gz`
