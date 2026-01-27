# Algebraic Rank Detection in dMRI

## Overview
This module implements a rigorous method for determining the number of fiber populations (Rank) in a voxel using tools from Algebraic Geometry, specifically the **Catalecticant Matrix** and the **Apolar Ideal**. 

Unlike traditional methods that rely on iterative fitting or heuristic thresholding of ODF peaks, this approach treats the signal as a symmetric tensor polynomial and uses algebraic properties to deduce the rank and directions directly.

## Theory

### The Signal as a Polynomial
The Diffusion MRI signal (at high b-values) is often modeled as a sum of ridge functions pointing along fiber directions $\mathbf{d}_i$:
$$ S(\mathbf{u}) = \sum_{i=1}^N w_i (\mathbf{u} \cdot \mathbf{d}_i)^L $$
where $L$ is an even integer (typically 4 or 6). This effectively represents the signal as a homogeneous polynomial of degree $L$ on the sphere.

### The Catalecticant Matrix
The **Catalecticant Matrix** (or Hankel matrix) $C$ is a matrix constructed from the coefficients of the polynomial. For a tensor of order $L=4$, the Catalecticant is a square matrix indexed by monomials of degree 2.
$$ C_{\alpha, \beta} = T_{\alpha+\beta} $$
where $T_\gamma$ are the coefficients of the tensor in the **Dual Integration** basis (Inverse Multinomial Scaling).

Key Property: **The rank of the Catalecticant matrix is equal to the number of fiber populations (tensor rank).**
- For $N=1$ fiber, $\text{rank}(C) = 1$.
- For $N=2$ fibers, $\text{rank}(C) = 2$.
- For $N=3$ fibers, $\text{rank}(C) = 3$.

### Inverse Multinomial Scaling
To ensure the Catalecticant has this rank property, the polynomial coefficients $c_\alpha$ (standard basis) must be converted to the tensor coefficients $T_\alpha$ by dividing by the multinomial coefficient:
$$ T_\alpha = \frac{c_\alpha}{\binom{L}{\alpha}} $$
This accounts for the geometry of the symmetric product space.

### Waring Decomposition (Root Finding)
Once the rank $r$ is determined, we can extract the unique fiber directions $\mathbf{d}_i$ by solving the **Waring Decomposition** problem.

#### Rank 1
The Catalecticant is rank 1: $C = \mathbf{v} \mathbf{v}^T$.
The vector $\mathbf{v}$ contains the monomials of degree $L/2$ evaluated at $\mathbf{d}$. 
$$ \mathbf{v} = [d_x^2, d_x d_y, d_x d_z, d_y^2, \dots]^T $$
The direction $\mathbf{d}$ is the principal eigenvector of $C$.

#### Rank 2 and 3
For $N > 1$, the kernel of the Catalecticant (the Null Space) contains polynomials that vanish at the fiber directions. This set of polynomials generates the **Apolar Ideal**.
To find the directions:
1.  Compute the Kernel of $C$.
2.  Select basis polynomials $Q_1, Q_2, \dots$ from the Kernel.
3.  Solve the system $Q_k(\mathbf{x}) = 0$ for $\mathbf{x} \in S^2$.
4.  This can be reduced to a Generalized Eigenvalue Problem (GEP) or simultaneous diagonalization of multiplication matrices.

## Implementation Details

### `dmipy_jax.rank_detection`
- **`construct_catalecticant`**:
    1.  Takes SH coefficients.
    2.  Applies Legendre scaling to map to Power Polynomial coefficients.
    3.  Maps to Monomial basis via Least Squares ($M$ matrix).
    4.  Applies Inverse Multinomial Scaling.
    5.  Fills the Hankel matrix.
- **`rank_determination`**: Computes SVD of $C$ and thresholds singular values.
- **`waring_decomposition_rank1`**: Extracts direction for the single fiber case.

### Calibration
The accuracy of the method depends on the precise definition of the Spherical Harmonic basis and the Monomial basis mapping. Discrepancies in normalization (Condon-Shortley phase, $4\pi$ factors) can break the rank structure. The `scaling` factors in `rank_detection.py` are critical for this alignment.
