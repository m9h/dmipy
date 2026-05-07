"""
SBI Simulator and Inference Pipeline for Morphogenesis.
Wraps the growth solver to enable parameter inference from folding patterns.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Optional, Tuple, Dict
from dmipy_jax.morphogenesis.cann import MorphoCANN
from dmipy_jax.morphogenesis.growth_solver import HyperelasticGrowthSolver
from dmipy_jax.morphogenesis.utils import get_growth_tensor

class MorphoSimulator(eqx.Module):
    """
    Wraps the morphogenesis pipeline into a standard SBI simulator:
    theta (parameters) -> x (summary statistics).
    """
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    n_iters: int = eqx.field(static=True)
    
    def __init__(self, grid_shape: Tuple[int, int, int] = (16, 16, 16), 
                 dx: float = 0.5, n_iters: int = 50):
        self.grid_shape = grid_shape
        self.dx = dx
        self.n_iters = n_iters

    def compute_summary_statistics(self, u: jax.Array) -> jax.Array:
        """
        Extracts summary statistics from the final displacement field.
        We use the mean and variance of the volumetric strain and curvature proxies.
        """
        # 1. Volumetric strain (J = det(F))
        grad_u0 = jnp.stack(jnp.gradient(u[0], self.dx), axis=0)
        grad_u1 = jnp.stack(jnp.gradient(u[1], self.dx), axis=0)
        grad_u2 = jnp.stack(jnp.gradient(u[2], self.dx), axis=0)
        grad_u = jnp.stack([grad_u0, grad_u1, grad_u2], axis=0)
        I = jnp.eye(3)[:, :, None, None, None]
        F = I + grad_u
        
        # Reshape to (N, 3, 3) for determinant
        N = jnp.prod(jnp.array(self.grid_shape))
        F_flat = F.transpose(2, 3, 4, 0, 1).reshape(N, 3, 3)
        J = jnp.linalg.det(F_flat)
        
        # 2. Simple 'folding index' based on local curvature proxy
        # Div(grad(u)) ~ Laplacian(u)
        lap_u = jnp.sum(jnp.stack([
            jnp.sum(jnp.stack(jnp.gradient(jnp.gradient(ui, self.dx)[i], self.dx), axis=0)[i], axis=0)
            for i, ui in enumerate(u)
        ]), axis=0)
        
        # Statistics
        stats = jnp.array([
            jnp.mean(J),
            jnp.std(J),
            jnp.mean(jnp.abs(lap_u)),
            jnp.std(lap_u),
            jnp.max(jnp.abs(u))
        ])
        
        return stats

    def __call__(self, params: jax.Array, key: jax.Array) -> jax.Array:
        """
        Runs a simulation given parameters.
        
        Args:
            params: [growth_ratio, w_iso_base, w_aniso_base]
            key: RNG key
            
        Returns:
            Summary statistics array.
        """
        growth_ratio = params[0]
        w_iso_val = params[1]
        w_aniso_val = params[2]
        
        # 1. Setup CANN with parameterized weights
        # We manually set the base weights for this run
        cann = MorphoCANN(n_basis=2, key=key)
        # Force the first weight to the parameter value for simplicity in this demo
        cann = eqx.tree_at(lambda m: m.w_iso, cann, jnp.full((2,), w_iso_val))
        cann = eqx.tree_at(lambda m: m.w_aniso, cann, jnp.full((2,), w_aniso_val))
        
        # 2. Solver
        solver = HyperelasticGrowthSolver(cann, self.grid_shape, self.dx)
        
        # 3. Setup Bilayer Growth
        Z = jnp.arange(self.grid_shape[2])
        is_gm = Z > (self.grid_shape[2] // 2)
        
        # Orientation (Radial)
        a0 = jnp.zeros((3,) + self.grid_shape)
        a0 = a0.at[2].set(1.0)
        
        # Growth Tensor
        g_map = jnp.where(is_gm, growth_ratio, 1.0)
        F_g = get_growth_tensor(g_map, a0, is_tangential=True)
        
        # 4. Solve
        u_final, _ = solver.solve_equilibrium(F_g, a0, n_iters=self.n_iters, lr=5e-4)
        
        # 5. Stats
        return self.compute_summary_statistics(u_final)

def generate_training_data(simulator: MorphoCANN, n_samples: int, key: jr.PRNGKey):
    """
    Generates a dataset of (theta, x) pairs for SBI training.
    """
    keys = jr.split(key, n_samples)
    
    # Priors
    # growth_ratio: [1.05, 1.25]
    # w_iso: [-1.0, 1.0] (log-space roughly)
    # w_aniso: [-1.0, 1.0]
    
    k1, k2, k3 = jr.split(key, 3)
    theta_growth = jr.uniform(k1, (n_samples,), minval=1.05, maxval=1.25)
    theta_w_iso = jr.uniform(k2, (n_samples,), minval=-1.0, maxval=1.0)
    theta_w_aniso = jr.uniform(k3, (n_samples,), minval=-1.0, maxval=1.0)
    
    thetas = jnp.stack([theta_growth, theta_w_iso, theta_w_aniso], axis=1)
    
    # Vectorized simulation (if possible, but solver uses scan, so we might need simple map or vmap)
    # For high-mem simulations, a loop or pmap is safer.
    sim_vmap = jax.vmap(simulator)
    xs = sim_vmap(thetas, keys)
    
    return thetas, xs
