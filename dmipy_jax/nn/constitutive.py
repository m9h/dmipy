import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple
from jaxtyping import Array, Float

class NonNegativeLinear(eqx.Module):
    """
    A linear layer that enforces non-negative weights and biases via softplus.
    Useful for convex potentials or enforcing physical constraints.
    """
    weight: Float[Array, "out_features in_features"]
    bias: Optional[Float[Array, "out_features"]]
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: jax.Array
    ):
        w_key, b_key = jax.random.split(key)
        # Initialize with smaller values so softplus'd values are reasonable
        self.weight = jax.random.normal(w_key, (out_features, in_features)) * 0.1
        if use_bias:
            self.bias = jax.random.normal(b_key, (out_features,)) * 0.1
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "in_features"]) -> Float[Array, "out_features"]:
        # Enforce non-negativity
        w_eff = jax.nn.softplus(self.weight)
        
        out = w_eff @ x
        
        if self.bias is not None:
            b_eff = jax.nn.softplus(self.bias)
            out = out + b_eff
            
        return out

class ConstitutiveNN(eqx.Module):
    """
    Bio-inspired Constitutive Neural Network (CANN) for Hyperelasticity.
    
    Architecture:
    Input: Invariants I1, I2
    Hidden: Split into 3 groups for Square, Exponential, and Logarithmic activations.
    Output: Strain Energy Density (Psi)
    """
    layer_in: NonNegativeLinear
    layer_out: NonNegativeLinear
    hidden_dim: int = eqx.field(static=True)
    
    def __init__(
        self,
        hidden_dim: int, # Should be divisible by 3 for equal splitting, or we handle remainder
        *,
        key: jax.Array
    ):
        key_in, key_out = jax.random.split(key)
        
        # We take 2 inputs: I1, I2
        # We output Psi (scalar)
        
        # Ensure hidden_dim is at least 3
        hidden_dim = max(hidden_dim, 3)
        self.hidden_dim = hidden_dim
        
        self.layer_in = NonNegativeLinear(2, hidden_dim, key=key_in)
        self.layer_out = NonNegativeLinear(hidden_dim, 1, use_bias=False, key=key_out)

    def __call__(self, I1: Float[Array, ""], I2: Float[Array, ""]) -> Float[Array, ""]:
        # 1. Input vector
        x = jnp.stack([I1, I2])
        
        # 2. Linear projection to hidden space
        # z shape: (hidden_dim,)
        z = self.layer_in(x)
        
        # 3. Split into 3 groups for different activation functions
        # We want to ensure we usage all neurons.
        # Let's split roughly equally.
        n = self.hidden_dim
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        
        # Indices for splitting using dynamic slicing is tricky in JAX if we want to be clean, 
        # but jnp.split works if sections are static, or we can slice.
        z_sq = z[:n1]
        z_exp = z[n1:n1+n2]
        z_log = z[n1+n2:]
        
        # 4. Apply activations
        # Square: x^2
        h_sq = jnp.square(z_sq)
        
        # Exp: e^x - 1 (to pass through origin? Or just exp?) 
        # LivingMatterLab usually uses (exp(x) - 1) or similar for zero-stress condition, 
        # but pure exp is requested. Let's stick to exp(x).
        h_exp = jnp.exp(z_exp)
        
        # Log: -log(x) or log(x)? 
        # For hyperelasticity (Neo-Hookean), terms like ln(J) appear. 
        # But here inputs z are outputs of SoftplusLinear * positive inputs. 
        # I1, I2 >= 3 for identity. z will be positive.
        # Let's use log(z).
        # Safety: add epsilon? z comes from softplus weights * inputs (>=3).
        # If bias is present and positive, z > 0.
        h_log = jnp.log(z_log) 
        
        # 5. Concatenate
        h = jnp.concatenate([h_sq, h_exp, h_log])
        
        # 6. Output projection (summation with weights)
        psi = self.layer_out(h)
        
        return psi[0] # Return scalar

    def get_stress(self, F: Float[Array, "3 3"]) -> Float[Array, "3 3"]:
        """
        Computes the First Piola-Kirchhoff stress tensor P = dPsi/dF.
        
        Args:
            F: Deformation Gradient (3x3)
            
        Returns:
            P: Stress Tensor (3x3)
        """
        
        def energy_fn(F_in):
            # Compute Invariants
            # Right Cauchy-Green Tensor C = F.T @ F
            C = F_in.T @ F_in
            
            # I1 = tr(C)
            I1 = jnp.trace(C)
            
            # I2 = 1/2 * ( (tr C)^2 - tr(C^2) )
            C2 = C @ C
            I2 = 0.5 * (I1**2 - jnp.trace(C2))
            
            # Compute Psi
            return self(I1, I2)
            
        # Differentiate energy w.r.t F
        P = jax.grad(energy_fn)(F)
        
        return P
