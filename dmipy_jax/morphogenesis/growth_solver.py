"""
JAX-based Hyperelastic Growth Solver for Brain Morphogenesis.
Implements the multiplicative decomposition F = Fe * Fg and energy minimization.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Callable, Tuple
from dmipy_jax.morphogenesis.cann import MorphoCANN

class HyperelasticGrowthSolver(eqx.Module):
    """
    Solver for growth-induced buckling in a 3D continuum.
    """
    
    cann: MorphoCANN
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    
    def __init__(self, cann: MorphoCANN, grid_shape: Tuple[int, int, int], dx: float = 1.0):
        self.cann = cann
        self.grid_shape = grid_shape
        self.dx = dx
        
    def get_deformation_gradient(self, u: jax.Array) -> jax.Array:
        """
        Calculates the deformation gradient F = I + grad(u).
        Uses central differences for the gradient.
        
        Args:
            u: Displacement field of shape (3, H, W, D).
            
        Returns:
            F: Deformation gradient field of shape (3, 3, H, W, D).
        """
        # We can use jnp.gradient or custom finite differences
        # For a 3D grid, jnp.gradient returns a list of 3 arrays (grad_h, grad_w, grad_d)
        # Each has shape (3, H, W, D)
        
        # We compute grad(u_i) for i in {0, 1, 2}
        grad_u0 = jnp.stack(jnp.gradient(u[0], self.dx), axis=0) # (3, H, W, D)
        grad_u1 = jnp.stack(jnp.gradient(u[1], self.dx), axis=0) # (3, H, W, D)
        grad_u2 = jnp.stack(jnp.gradient(u[2], self.dx), axis=0) # (3, H, W, D)
        
        grad_u = jnp.stack([grad_u0, grad_u1, grad_u2], axis=0) # (3, 3, H, W, D)
        
        # F = I + grad_u
        I = jnp.eye(3)[:, :, None, None, None]
        return I + grad_u

    def total_energy(self, u: jax.Array, F_g: jax.Array, a0: Optional[jax.Array] = None, 
                     mask: Optional[jax.Array] = None) -> jax.Array:
        """
        Calculates the total elastic energy of the system.
        
        Args:
            u: Displacement field (3, H, W, D).
            F_g: Growth deformation gradient field (3, 3, H, W, D).
            a0: Fiber orientation field (3, H, W, D).
            mask: Optional tissue mask (H, W, D).
            
        Returns:
            Scalar energy.
        """
        F = self.get_deformation_gradient(u)
        
        # F = Fe * Fg  => Fe = F * inv(Fg)
        # Assuming F_g is diagonal or easily invertible for performance
        # Here we do a general inverse per voxel
        # Reshape to (H*W*D, 3, 3) for vectorized inverse
        h, w, d = self.grid_shape
        N = h * w * d
        F_flat = F.transpose(2, 3, 4, 0, 1).reshape(N, 3, 3)
        Fg_flat = F_g.transpose(2, 3, 4, 0, 1).reshape(N, 3, 3)
        
        # Vectorized inverse
        Fg_inv_flat = jnp.linalg.inv(Fg_flat)
        Fe_flat = jnp.matmul(F_flat, Fg_inv_flat)
        
        if a0 is not None:
            a0_flat = a0.transpose(1, 2, 3, 0).reshape(N, 3)
        else:
            a0_flat = None
            
        # Vmap the CANN psi function over the grid
        # self.cann.psi(F, a0)
        # We need to handle a0_flat=None in vmap correctly
        if a0_flat is not None:
            vmap_psi = jax.vmap(self.cann.psi)
            psi_vals = vmap_psi(Fe_flat, a0_flat)
        else:
            # We wrap the call to handle the optional a0
            vmap_psi = jax.vmap(lambda f: self.cann.psi(f, None))
            psi_vals = vmap_psi(Fe_flat)
            
        if mask is not None:
            psi_vals = psi_vals * mask.reshape(-1)
            
        return jnp.sum(psi_vals) * (self.dx ** 3)

    def solve_equilibrium(self, F_g: jax.Array, a0: Optional[jax.Array] = None, 
                          mask: Optional[jax.Array] = None,
                          n_iters: int = 100, lr: float = 1e-3) -> jax.Array:
        """
        Finds the equilibrium displacement field u by minimizing energy.
        Uses Gradient Descent (Gradient Flow) as a simple solver.
        
        In Phase 3, we would use more robust solvers (L-BFGS via jaxopt).
        """
        
        u = jnp.zeros((3,) + self.grid_shape)
        
        def loss_fn(u_current):
            return self.total_energy(u_current, F_g, a0, mask)
        
        grad_fn = jax.grad(loss_fn)
        
        # Simple GD loop
        # We use a scan for efficiency and differentiability
        def step_fn(u_i, _):
            g = grad_fn(u_i)
            # Boundary conditions: fix the bottom layer or a central core
            # For simplicity, we just zero the gradient at boundaries if needed
            u_next = u_i - lr * g
            return u_next, loss_fn(u_next)
            
        u_final, history = jax.lax.scan(step_fn, u, None, length=n_iters)
        return u_final, history
