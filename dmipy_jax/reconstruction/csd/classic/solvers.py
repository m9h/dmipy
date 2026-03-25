
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import jax.scipy.optimize

from dmipy_jax.utils.spherical_harmonics import sh_basis_real
from dmipy_jax.reconstruction.csd.classic.algebraic import precompute_sos_mapping, build_sos_tensor

def _get_response_matrix(response, lmax):
    """
    Constructs the diagonal Convolution Matrix R.
    Args:
        response: Array of SH coefficients of the response function (zonal).
                  Expected shape: (lmax/2 + 1,) for l=0, 2, ... lmax.
                  Values: [R0, R2, R4, ...]
        lmax: Maximum SH order.
    Returns:
        R_diag vector of shape (N_coeffs,) such that S_lm = R_diag[lm] * F_lm
    """
    # Map L-indices to LM-indices
    # Coefficients are ordered: 0,0, 2,-2, 2,-1...
    # We populate R_diag such that for each (l,m), value is response[l/2].
    
    # N_coeffs for symmetric real SH:
    # l=0: 1 coeff
    # l=2: 5 coeffs
    # l=4: 9 coeffs
    # ...
    
    r_full = []
    
    # Check if response matches lmax
    if len(response) != (lmax // 2 + 1):
        # Fallback or error?
        pass

    idx = 0
    for l_idx, l in enumerate(range(0, lmax + 1, 2)):
        n_m = 2 * l + 1
        # Append R_l repeated n_m times
        r_val = response[l_idx]
        r_full.extend([r_val] * n_m)
        
    return jnp.array(r_full)

def fit_sos_csd(data, bvecs, bvals, response, lmax_fod=8, lmax_root=4, lambda_reg=0.0):
    """
    Fits Algebraic SOS-CSD to dMRI data.
    
    Args:
        data: (N_vox, N_dirs)
        bvecs: (N_dirs, 3)
        bvals: (N_dirs,) - Used to filter b0/shell if needed? 
               Assumption: Data is single shell or appropriate shell for CSD.
        response: Response function coefficients [R0, R2, R4, R6, R8]
        lmax_fod: Target FOD SH order (default 8).
        lmax_root: Root polynomial order (default 4). 2*lmax_root must >= lmax_fod usually.
        lambda_reg: Regularization on a_coeffs norm (Tikhonov).
        
    Returns:
        fod_coeffs: (N_vox, N_sh)
    """
    
    # 1. Precompute Algebra
    Y_r, Y_f, w = precompute_sos_mapping(lmax_root, lmax_fod)
    G_tensor = build_sos_tensor(Y_r, Y_f, w) # (45, 15, 15)
    
    # 2. Precompute Acquisition Matrix
    # Convert bvecs to theta, phi
    # dmipy_jax utils?
    # Simple conversion:
    r = jnp.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-8
    bvecs_norm = bvecs / r
    theta = jnp.arccos(bvecs_norm[:, 2]) # 0 to pi
    phi = jnp.arctan2(bvecs_norm[:, 1], bvecs_norm[:, 0])
    
    Y_acq = sh_basis_real(theta, phi, lmax=lmax_fod) # (N_dirs, 45)
    
    # 3. Response Convolution
    R_diag = _get_response_matrix(response, lmax_fod)
    
    # Effective Observation Matrix: K = Y_acq @ diag(R)
    # S = K @ c_fod
    K = Y_acq * R_diag[None, :] # element-wise broadcast: Y_lm * R_l
    
    # 4. Define Loss Function
    # Params a: (15,)
    @jit
    def loss_fn(a, signal_meas):
        # 1. SOS Mapping: a -> c_fod
        # c_fod_k = sum(G_kij * ai * aj)
        # Efficient: aG = tensordot(a, G, axes=1) -> (15, 45) ??
        # G is (45, 15, 15).
        # c_fod = einsum('kij, i, j -> k', G_tensor, a, a)
        c_fod = jnp.einsum('kij,i,j->k', G_tensor, a, a)
        
        # 2. Forward Model
        pred_signal = jnp.dot(K, c_fod)
        
        # 3. Residual
        res = pred_signal - signal_meas
        mse = jnp.sum(res**2)
        
        reg = lambda_reg * jnp.sum(a**2)
        
        return mse + reg

    # 5. Optimization Loop (Vmap over voxels)
    # Using BFGS from jax.scipy.optimize
    
    def solve_voxel(signal):
        # Init: a=0 is bad (stationary).
        # Random init or Heuristic?
        # Heuristic: Match S0?
        a0 = jnp.ones(G_tensor.shape[1]) * 0.1 
        
        res = jax.scipy.optimize.minimize(
            loss_fn, a0, args=(signal,), method='BFGS',
            options={'maxiter': 50}
        )
        
        a_opt = res.x
        # Compute final FOD
        c_fod = jnp.einsum('kij,i,j->k', G_tensor, a_opt, a_opt)
        return c_fod

    print("Running SOS-CSD Optimization...")
    fod_coeffs = vmap(solve_voxel)(data)
    

def _get_sphere_projection(lmax, n_theta=None, n_phi=None):
    """
    Returns matrix P to map SH -> Sphere and P_inv for Sphere -> SH.
    Uses a standard grid (e.g. Fibonacci).
    """
    # 1. Grid
    n_points = 200 # Enough for L=8
    key = jax.random.key(0)
    golden = (1 + 5**0.5)/2
    i = jnp.arange(n_points)
    phi = 2 * jnp.pi * (i / golden % 1)
    costheta = 1 - 2*(i + 0.5)/n_points
    theta = jnp.arccos(costheta)
    
    weights = jnp.full((n_points,), 4 * jnp.pi / n_points)
    
    Y_sphere = sh_basis_real(theta, phi, lmax=lmax) # (N_p, N_sh)
    
    # Pseudo-inverse? 
    # Or least squares matrix.
    # Map Coeffs -> Amps: A = Y * c
    # Map Amps -> Coeffs: c = pinv(Y) @ A (or weighted integration)
    
    # Weighted integration is better if grid is good.
    # c_lm = sum( w_p * f(p) * Y_lm(p) )
    # But for least squares fit to samples:
    Y_inv = jnp.linalg.pinv(Y_sphere)
    
    return Y_sphere, Y_inv


def fit_csd(data, bvecs, bvals, response, lmax=8, lambda_reg=1.0, max_iter=50):
    """
    Classic Iterative CSD (Tournier 2007).
    Uses Iterative Soft-Thresholding.
    
    Args:
        data: (N_vox, N_dirs)
        bvecs: (N_dirs, 3)
        bvals: (N_dirs,)
        response: Response coeffs [R0, R2...]
        lmax: Order.
        lambda_reg: Regularization weight.
        
    Returns:
        fod_coeffs: (N_vox, N_sh)
    """
    
    # 1. Setup Matrices
    # Acq Matrix
    r = jnp.linalg.norm(bvecs, axis=1, keepdims=True) + 1e-8
    bvecs_norm = bvecs / r
    theta_acq = jnp.arccos(bvecs_norm[:, 2]) 
    phi_acq = jnp.arctan2(bvecs_norm[:, 1], bvecs_norm[:, 0])
    
    Y_acq = sh_basis_real(theta_acq, phi_acq, lmax=lmax)
    R_diag = _get_response_matrix(response, lmax)
    K = Y_acq * R_diag[None, :] # Convolution Matrix
    
    # Precompute K inverse (K^T K + lambda I)^-1 K^T not constant because lambda usually fixed? 
    # Tournier 2007: Constrained minimization.
    # Regularized Least Squares: (K^T K + lambda L^T L)^-1 K^T y
    # Here we solve: min ||Kc - y||^2  s.t. FOD > 0
    # Iterative approach:
    # 1. Initial LS fit.
    # 2. Loop:
    #    - Update directions (negative lobes)
    #    - Solve Regularized LS where negative directions are penalized heavily.
    
    # Precompute standard pseudo-inverse for init
    K_pinv = jnp.linalg.pinv(K)
    
    # Sphere Projection
    Y_sphere, Y_sphere_inv = _get_sphere_projection(lmax)
    
    # Vmap function
    def solve_voxel(signal):
        # Init: Regular LS
        c_curr = jnp.dot(K_pinv, signal)
        
        # Iteration state: c
        def body_fn(i, c):
            # 1. Evaluate on Sphere
            amp = jnp.dot(Y_sphere, c)
            
            # 2. Identify Negative Lobes
            # Soft thresholding loop logic in CSD:
            # We want to minimize ||Kc - y||^2 + lambda ||H c||^2 
            # where H penalizes negative regions.
            # In simple iterative alg:
            # We assume negative regions should be zero.
            # We estimate amplitudes, clip negatives to zero, project back?
            # That's POCS. Tournier uses slightly different logic (weighting matrix).
            
            # Simple Projected Gradient Descent logic:
            # grad = 2 K.T (K c - y).
            # c = c - alpha * grad
            # Project: c -> amp -> clip(amp, 0, inf) -> c_proj
            
            # Let's implement Projected Gradient Descent (PGD)
            # 1. Gradient step
            pred = jnp.dot(K, c)
            res = pred - signal
            grad = jnp.dot(K.T, res)
            
            # Step size? 
            # Lipschitz constant L ~ max eig(K.T K)
            # Approx: 1.0 / max(R)^2 ?
            step = 1e-3 # Conservative
            c_desc = c - step * grad
            
            # 2. Projection
            amp_desc = jnp.dot(Y_sphere, c_desc)
            amp_proj = jnp.maximum(amp_desc, 0.0) # Clip
            c_proj = jnp.dot(Y_sphere_inv, amp_proj)
            
            return c_proj

        # Run Scan
        # For convergence, 50-100 iters.
        c_final = jax.lax.fori_loop(0, max_iter, body_fn, c_curr)
        
        return c_final

    print("Agent 1: Running Classic iCSD (PGD)...")
    fod_coeffs = vmap(solve_voxel)(data)
    
    return fod_coeffs
