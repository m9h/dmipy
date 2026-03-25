import jax
import jax.numpy as jnp
import jax.scipy.optimize
from dmipy_jax.acquisition import JaxAcquisition

def aganj_gradient_cost(params, log_signal, bvecs, bvals, tau=0.04):
    """
    Cost function for Aganj Gradient Estimator.
    Minimizes: sum( (logS + qDq*tau - 0.5*log(1 + (qDg*tau)^2) )^2 )
    
    Params: [D_xx, D_xy, D_xz, D_yy, D_yz, D_zz, g_x, g_y, g_z] (9,)
    """
    # Unpack params
    # D elements
    D_xx, D_xy, D_xz, D_yy, D_yz, D_zz = params[:6]
    # g vector
    g = params[6:]
    
    # Construct D tensor (re-assembly for vec ops)
    # But explicitly writing quadratic form is faster/cleaner.
    
    # q vector
    # q = sqrt(b / tau) * u
    q_mag = jnp.sqrt(bvals / tau)
    qx = q_mag * bvecs[:, 0]
    qy = q_mag * bvecs[:, 1]
    qz = q_mag * bvecs[:, 2]
    
    # Compute q.T @ D @ q
    # = Dxx qx^2 + Dyy qy^2 + Dzz qz^2 + 2Dxy qx qy + ...
    qDq = (D_xx * qx**2 + D_yy * qy**2 + D_zz * qz**2 + 
           2 * (D_xy * qx * qy + D_xz * qx * qz + D_yz * qy * qz))
    
    # Compute q.T @ D @ g
    # D @ g = [Dxx gx + Dxy gy + Dxz gz, ...]
    Dgx = D_xx * g[0] + D_xy * g[1] + D_xz * g[2]
    Dgy = D_xy * g[0] + D_yy * g[1] + D_yz * g[2]
    Dgz = D_xz * g[0] + D_yz * g[1] + D_zz * g[2]
    
    qDg = qx * Dgx + qy * Dgy + qz * Dgz
    
    # Terms
    # logS is negative.
    # Model: logS = - qDq * tau + 0.5 * log(1 + (tau * qDg)^2)
    # Residual = logS + qDq*tau - 0.5...
    
    term_qDq = qDq * tau
    term_asym = 0.5 * jnp.log(1 + (tau * qDg)**2)
    
    residual = log_signal + term_qDq - term_asym
    
    return jnp.sum(residual**2)

def fit_aganj_gradients(signal, acquisition, tau=0.04):
    """
    Fits D and g to estimate spatial gradients.
    
    Args:
        signal: S/S0 (normalized).
    Weights: DTI fit for initialization.
    """
    # 1. Initialize D from standard DTI (approx)
    # logS ~ -b uDu
    # -logS/b ~ uDu
    log_s = jnp.log(jnp.clip(signal, 1e-6, 1.0))
    
    # Quick DTI Linear Least Squares
    bvecs = acquisition.gradient_directions
    bvals = acquisition.bvalues
    
    # Design matrix: [x^2, y^2, z^2, 2xy, 2xz, 2yz]
    X = jnp.stack([
        bvecs[:,0]**2, bvecs[:,1]**2, bvecs[:,2]**2,
        2*bvecs[:,0]*bvecs[:,1], 2*bvecs[:,0]*bvecs[:,2], 2*bvecs[:,1]*bvecs[:,2]
    ], axis=-1)
    
    # Weighted by bvals
    # ln(S) = ln(S0) - b * uDu
    # Y = -ln(S/S0) / b  (approx ADC?)
    # or just fit to -ln(S/S0).
    # Y = -log_s
    # X_scaled = X * bvals[:, None]
    
    # We want D elements.
    # X_scaled @ d = -log_s
    # d = pinv(X_scaled) @ (-log_s)
    
    X_scaled = X * bvals[:, None]
    d_init = jnp.linalg.pinv(X_scaled) @ (-log_s)
    
    # Init params: [d_init, g_noise]
    # Break symmetry at g=0
    # Use deterministic noise or just constant small value
    g_init = jnp.array([0.01, 0.01, 0.01])
    p0 = jnp.concatenate([d_init, g_init])
    
    # 2. Optimize
    # Use jax.scipy.optimize.minimize (BFGS)
    # vmap this function over voxels outside.
    
    def cost(p):
        return aganj_gradient_cost(p, log_s, bvecs, bvals, tau)
        
    res = jax.scipy.optimize.minimize(cost, p0, method='BFGS', tol=1e-5, options={'maxiter': 500})
    
    return res.x # [D..., g...]

