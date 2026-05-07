"""
Constitutive Artificial Neural Networks (CANNs) for brain tissue mechanics.
Following Ellen Kuhl's "Constitutive artificial neural networks: A paradigm shift 
in modeling soft tissue" (2023).
"""

from __future__ import annotations
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, List

class MorphoCANN(eqx.Module):
    """
    A Constitutive Artificial Neural Network for hyperelastic brain tissue.
    
    This module represents the strain energy density function Psi(I1, J, I4)
    where:
    - I1 is the first invariant of the right Cauchy-Green tensor C = F^T F.
    - J = det(F) is the volume ratio.
    - I4 = a0 . C . a0 is the squared stretch along the fiber direction a0.
    
    The architecture ensures polyconvexity by using non-negative weights and 
    monotonic basis functions.
    """
    
    # Weights for isotropic and anisotropic terms
    w_iso: jax.Array  # Weights for (I1 - 3) terms
    w_vol: jax.Array  # Weights for (J - 1)^2 terms
    w_aniso: jax.Array # Weights for (I4 - 1)^2 terms
    
    # Exponents (learned)
    exp_iso: jax.Array
    exp_vol: jax.Array
    exp_aniso: jax.Array
    
    n_basis: int = eqx.field(static=True)
    
    def __init__(self, n_basis: int = 4, *, key: jax.Array):
        kw, ke = jr.split(key)
        self.n_basis = n_basis
        
        # Initialize weights and exponents in log-space for positivity
        self.w_iso = jr.normal(kw, (n_basis,))
        self.w_vol = jr.normal(jr.split(kw)[0], (n_basis,))
        self.w_aniso = jr.normal(jr.split(kw)[1], (n_basis,))
        
        self.exp_iso = jr.normal(ke, (n_basis,))
        self.exp_vol = jr.normal(jr.split(ke)[0], (n_basis,))
        self.exp_aniso = jr.normal(jr.split(ke)[1], (n_basis,))
        
    def psi(self, F: jax.Array, a0: Optional[jax.Array] = None) -> jax.Array:
        """
        Calculates the strain energy density Psi.
        
        Args:
            F: Deformation gradient tensor (3, 3).
            a0: Initial fiber direction vector (3,). If None, model is isotropic.
            
        Returns:
            Scalar strain energy density.
        """
        # Right Cauchy-Green tensor C
        C = jnp.dot(F.T, F)
        
        # Invariants
        I1 = jnp.trace(C)
        J = jnp.linalg.det(F)
        
        # Isotropic terms (Neo-Hookean and higher order)
        # Using (I1 - 3) to ensure Psi=0 at F=I
        x_iso = jnp.maximum(I1 - 3.0, 0.0)
        w_iso = jax.nn.softplus(self.w_iso)
        # Exponents between 1 and 2 for stability
        e_iso = 1.0 + jax.nn.sigmoid(self.exp_iso)
        psi_iso = jnp.sum(w_iso * (x_iso ** e_iso))
        
        # Volumetric terms (Bulk response)
        # Using (J - 1)^2 or similar
        x_vol = (J - 1.0)**2
        w_vol = jax.nn.softplus(self.w_vol)
        e_vol = 1.0 + jax.nn.sigmoid(self.exp_vol)
        psi_vol = jnp.sum(w_vol * (x_vol ** e_vol))
        
        # Anisotropic terms (Axonal tension/stiffness)
        psi_aniso = 0.0
        if a0 is not None:
            # I4 is the squared stretch along the fiber
            I4 = jnp.dot(a0, jnp.dot(C, a0))
            # Only consider tension (I4 > 1)
            x_aniso = jnp.maximum(I4 - 1.0, 0.0)
            w_aniso = jax.nn.softplus(self.w_aniso)
            e_aniso = 1.0 + jax.nn.sigmoid(self.exp_aniso)
            psi_aniso = jnp.sum(w_aniso * (x_aniso ** e_aniso))
            
        return psi_iso + psi_vol + psi_aniso

    def first_piola_stress(self, F: jax.Array, a0: Optional[jax.Array] = None) -> jax.Array:
        """
        Calculates the First Piola-Kirchhoff stress tensor P = dPsi/dF.
        """
        return jax.grad(self.psi)(F, a0)

    def cauchy_stress(self, F: jax.Array, a0: Optional[jax.Array] = None) -> jax.Array:
        """
        Calculates the Cauchy stress tensor sigma = (1/J) * P * F^T.
        """
        J = jnp.linalg.det(F)
        P = self.first_piola_stress(F, a0)
        return (1.0 / J) * jnp.dot(P, F.T)

class BrainMorphologyState(eqx.Module):
    """
    Represents the state of a developing brain region.
    """
    F_g: jax.Array  # Growth deformation gradient
    F_e: jax.Array  # Elastic deformation gradient
    a0: jax.Array   # Fiber orientation
    
    def total_deformation(self) -> jax.Array:
        return jnp.dot(self.F_e, self.F_g)
