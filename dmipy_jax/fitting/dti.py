import jax
import jax.numpy as jnp
from functools import partial

def fit_dti(data: jnp.ndarray, bvals: jnp.ndarray, bvecs: jnp.ndarray, min_signal: float = 1e-6):
    """
    Fits DTI using Linear Least Squares on log-signal.
    
    Args:
        data: (N_meas,) Signal.
        bvals: (N_meas,) b-values.
        bvecs: (N_meas, 3) gradients.
        
    Returns:
        eigvals: (3,) eigenvalues (sorted ascending? or descending? usually e3>e2>e1 in standard convention, but eigh returns asc).
                 We want e1 as primary? Usually e1 >= e2 >= e3 is primary.
                 jnp.linalg.eigh returns ascending. So e[2] is primary.
        eigvecs: (3, 3) eigenvectors. col[:, i] corresponds to eigval[i].
        meas_S0: Estimated S0.
    """
    # approx: ln(S) = ln(S0) - b * gT D g
    # Design matrix X: [1, -b*gx^2, -b*gy^2, -b*gz^2, -2b*gx*gy, -2b*gx*gz, -2b*gy*gz]
    # Coeffs beta: [ln(S0), Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    
    # Avoid log(0)
    data_safe = jnp.maximum(data, min_signal)
    Y = jnp.log(data_safe)
    
    bx, by, bz = bvecs.T * jnp.sqrt(bvals) # weighted bvecs? No.
    # Term is -b * (g . D . g)
    # = -b * (gx^2 Dxx + ... + 2 gx gy Dxy ...)
    # Let's construct design matrix rows.
    
    # X shape (N, 7)
    # Col 0: 1
    # Col 1: -b * gx^2
    # ...
    gx, gy, gz = bvecs.T
    
    # Precompute design matrix columns
    # Note: bvals has shape (N,)
    X = jnp.stack([
        jnp.ones_like(bvals),
        -bvals * gx * gx,
        -bvals * gy * gy,
        -bvals * gz * gz,
        -2 * bvals * gx * gy,
        -2 * bvals * gx * gz,
        -2 * bvals * gy * gz
    ], axis=1)
    
    # Solve X beta = Y
    # beta = pinv(X) Y
    beta = jnp.linalg.lstsq(X, Y, rcond=None)[0]
    
    # Extract Tensor parameters
    ln_S0 = beta[0]
    Dxx, Dyy, Dzz = beta[1], beta[2], beta[3]
    Dxy, Dxz, Dyz = beta[4], beta[5], beta[6]
    
    # Form Tensor matrix (3, 3) - Symmetric
    D = jnp.array([
        [Dxx, Dxy, Dxz],
        [Dxy, Dyy, Dyz],
        [Dxz, Dyz, Dzz]
    ])
    
    # Eigendecomposition
    # eigh for hermitian (symmetric real)
    evals, evecs = jnp.linalg.eigh(D)
    
    # eigh returns ascending order: e[0]<=e[1]<=e[2].
    # Primary direction V1 is evecs[:, 2].
    
    return evals, evecs, jnp.exp(ln_S0)

def compute_fa_md(evals):
    """
    Computes FA and MD from eigenvalues.
    evals: (3,) or (..., 3)
    """
    # MD = mean(evals)
    md = jnp.mean(evals, axis=-1)
    
    # FA = sqrt(3/2) * sqrt( (e - md)^2 ) / sqrt( e^2 )
    # Variance of evals
    
    # Deviation from mean
    # dev = evals - md[..., None]
    # sum_sq_diff = sum(dev^2)
    # but more stable formula:
    
    e1, e2, e3 = evals[..., 0], evals[..., 1], evals[..., 2]
    # Note: assumed last dim is 3.
    
    num = (e1 - md)**2 + (e2 - md)**2 + (e3 - md)**2
    denom = e1**2 + e2**2 + e3**2
    
    # Safety for denom=0 (empty voxel)
    fa = jnp.sqrt(1.5 * num / (denom + 1e-9))
    
    return fa, md
