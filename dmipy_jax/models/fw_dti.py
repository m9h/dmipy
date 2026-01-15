"""
Spatially Regularized Free Water DTI Model.

Proposed algorithm for fitting a Free Water DTI model (Tensor + Ball) using
spatial regularization on the free water fraction map.
"""

import jax
import jax.numpy as jnp
import jaxopt
from functools import partial

def _cholesky_to_tensor(c_params):
    """
    Constructs Symmetric Positive Definite Tensor D from Cholesky parameters.
    
    Args:
        c_params: Array of shape (..., 6)
                  [c1, c2, c3, c4, c5, c6]
                  
    Returns:
        D: Array of shape (..., 3, 3)
    """
    # L = [[c1, 0,  0 ],
    #      [c2, c3, 0 ],
    #      [c4, c5, c6]]
    
    c1 = c_params[..., 0]
    c2 = c_params[..., 1]
    c3 = c_params[..., 2]
    c4 = c_params[..., 3]
    c5 = c_params[..., 4]
    c6 = c_params[..., 5]
    
    zeros = jnp.zeros_like(c1)
    
    # Stack to create (..., 3, 3) matrix L
    # Row 0: [c1, 0, 0]
    r0 = jnp.stack([c1, zeros, zeros], axis=-1)
    # Row 1: [c2, c3, 0]
    r1 = jnp.stack([c2, c3, zeros], axis=-1)
    # Row 2: [c4, c5, c6]
    r2 = jnp.stack([c4, c5, c6], axis=-1)
    
    L = jnp.stack([r0, r1, r2], axis=-2)
    
    # D = L @ L.T
    D = jnp.matmul(L, jnp.swapaxes(L, -1, -2))
    return D

def fw_dti_predict(params_grid, bvals, bvecs, d_iso=3.0e-9):
    """
    Predicts signal for a grid of voxels.
    
    Args:
        params_grid: (X, Y, Z, 8)
                     0: S0
                     1: f_iso (free water fraction)
                     2-7: Tensor Cholesky params (c1..c6)
        bvals: (B,)
        bvecs: (B, 3)
        d_iso: Isotropic diffusivity (default 3.0e-9 m^2/s = 3.0 um^2/ms)
        
    Returns:
        S_pred: (X, Y, Z, B)
    """
    S0 = params_grid[..., 0:1] # (X, Y, Z, 1) keeps dims for broadcasting
    f_iso = params_grid[..., 1:2] # (X, Y, Z, 1)
    c_params = params_grid[..., 2:]
    
    # Ensure f_iso is within [0, 1] if not strictly bounded by optimizer, 
    # generally handled by bounds in L-BFGS-B, but here we just use it.
    
    # 1. Isotropic Signal
    # E_iso = exp(-b * D_iso)
    E_iso = jnp.exp(-bvals * d_iso) # (B,)
    
    # 2. Tensor Signal
    D = _cholesky_to_tensor(c_params) # (X, Y, Z, 3, 3)
    
    # Calculate b * g^T D g
    # einsum: 
    # D: ...ij
    # bvecs: bj, bi (shared index b)
    # contraction: ...ij, bj, bi -> ...b
    bDg = jnp.einsum('...ij, bj, bi -> ...b', D, bvecs, bvecs)
    
    E_aniso = jnp.exp(-bvals * bDg) # (X, Y, Z, B)
    
    # Combine
    # S = S0 * [f_iso * E_iso + (1-f_iso) * E_aniso]
    # Broadcast E_iso to (X, Y, Z, B)
    
    S_pred = S0 * (f_iso * E_iso + (1.0 - f_iso) * E_aniso)
    return S_pred

def loss_function(params_flat, data, bvals, bvecs, mask, lambda_reg, shape, d_iso):
    """
    Loss function for block-based optimization.
    
    Args:
        params_flat: 1D array of parameters (X*Y*Z*8,)
        data: (X, Y, Z, B)
        bvals, bvecs: global acquisition
        mask: (X, Y, Z)
        lambda_reg: regularization weight
        shape: (X, Y, Z)
        d_iso: float
    """
    # Reshape parameters
    params_grid = params_flat.reshape(shape + (8,))
    
    # Forward pass
    S_pred = fw_dti_predict(params_grid, bvals, bvecs, d_iso=d_iso)
    
    # Data Mismatch (MSE)
    # Only inside mask
    resid = S_pred - data
    mse_map = jnp.mean(resid**2, axis=-1) # Mean over B-values
    
    # Weighted by mask
    data_loss = jnp.sum(mse_map * mask) / (jnp.sum(mask) + 1e-9)
    
    # Regularization
    # Beltrami / TV on f_iso
    f_iso = params_grid[..., 1]
    
    # Gradient of f_iso
    grads = jnp.gradient(f_iso) # Tuple of arrays (one per dim)
    
    # Calculate mean absolute gradient magnitude sum
    # "mean(abs(gradient(f_iso)))"
    # We can perform mean over the mask or whole volume?
    # Usually regularization should encourage smoothness everywhere or just in tissue.
    # Pasternak's method propagates information so global smoothness is often desired.
    # We will compute mean over the whole block to avoiding boundary artifacts with mask.
    
    reg_loss = 0.0
    for g in grads:
        reg_loss += jnp.mean(jnp.abs(g))
        
    total_loss = data_loss + lambda_reg * reg_loss
    return total_loss


class SpatiallyRegularizedFWDTI:
    def __init__(self, data, bvals, bvecs, mask, d_iso=3.0e-9):
        """
        Initialize the fit configuration.
        
        Args:
            data: (X, Y, Z, B)
            bvals: (B,)
            bvecs: (B, 3)
            mask: (X, Y, Z) boolean or float mask
            d_iso: fixed isotropic diffusivity (default 3.0e-9 m^2/s)
        """
        self.data = data
        self.bvals = bvals
        self.bvecs = bvecs
        self.mask = mask
        self.d_iso = d_iso
        self.shape = data.shape[:-1]
        self.n_voxels = int(jnp.prod(jnp.array(self.shape)))
        
    def fit(self, lambda_reg=0.1, x0=None):
        """
        Run the block-based optimization using L-BFGS-B.
        
        Args:
            lambda_reg: Regularization weight for TV on f_iso.
            x0: Initial guess (flat array). If None, generated automatically.
            
        Returns:
            OptimizationResult from jaxopt
        """
        # Define bounds
        # 8 params per voxel: S0, f_iso, c1..c6
        # S0: [0, inf)
        # f_iso: [0, 1]
        # c: (-inf, inf)
        
        lower_bounds_voxel = jnp.array([0.0, 0.0] + [-jnp.inf]*6)
        upper_bounds_voxel = jnp.array([jnp.inf, 1.0] + [jnp.inf]*6)
        
        # Tile bounds
        # shape: (N_voxels * 8,)
        lower_bounds = jnp.tile(lower_bounds_voxel, self.n_voxels)
        upper_bounds = jnp.tile(upper_bounds_voxel, self.n_voxels)
        
        bounds = (lower_bounds, upper_bounds)
        
        # Initial Guess
        if x0 is None:
            # Simple initialization
            # S0 ~ mean(data[..., b0_indices]) or just max(data)
            # We'll use a rough constant for robustness or per-voxel max if cheap.
            # Let's use max of data per voxel.
            param_grid_init = jnp.zeros(self.shape + (8,))
            
            # S0 guess: max signal
            s0_guess = jnp.max(self.data, axis=-1)
            param_grid_init = param_grid_init.at[..., 0].set(s0_guess)
            
            # f_iso guess: 0.5
            param_grid_init = param_grid_init.at[..., 1].set(0.5)
            
            # Tensor guess: sphere (c1=c3=c6=sqrt(diff), others 0)
            # diff ~ 1e-9 (approx mean diffusivity for brain)
            diff_guess = 1.0e-9
            c_val = jnp.sqrt(diff_guess)
            param_grid_init = param_grid_init.at[..., 2].set(c_val) # c1
            param_grid_init = param_grid_init.at[..., 4].set(c_val) # c3
            param_grid_init = param_grid_init.at[..., 7].set(c_val) # c6
            
            x0 = param_grid_init.ravel()
        
        # Setup Optimizer
        optimizer = jaxopt.ScipyMinimize(
            fun=loss_function,
            method='L-BFGS-B'
        )
        
        # Run
        # partial args
        # loss_function(params_flat, data, bvals, bvecs, mask, lambda_reg, shape, d_iso)
        
        res = optimizer.run(
            x0,
            bounds=bounds,
            data=self.data,
            bvals=self.bvals,
            bvecs=self.bvecs,
            mask=self.mask,
            lambda_reg=lambda_reg,
            shape=self.shape,
            d_iso=self.d_iso
        )
        
        return res
