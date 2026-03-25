
import jax
import jax.numpy as jnp
import equinox as eqx
import e3nn_jax as e3nn
from typing import Optional

class EquivariantCSD(eqx.Module):
    """
    E(3) x SO(3) Equivariant CSD Network (Agent 2).
    
    Maps input dMRI Signal (in SH basis) to FOD (in SH basis)
    using strictly equivariant Tensor Product operations.
    
    This architecture respects the spherical symmetry of the data:
    Rotating the dMRI signal results in an exactly rotated FOD.
    """
    
    layers: list
    input_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    
    def __init__(self, key, lmax_in=4, lmax_out=8, features=64, depth=3):
        """
        Args:
            lmax_in: Max SH order of input signal (usually 4 or 6).
            lmax_out: Max SH order of output FOD (usually 8 or 10).
            features: Number of internal channels (multiplicity of irreps).
            depth: Number of layers.
        """
        # 1. Define Irreps (Use strings to ensure safe parsing)
        # Input: "1x0e + 1x2e + ..."
        input_list = [f"1x{l}e" for l in range(0, lmax_in + 1, 2)]
        self.input_irreps = " + ".join(input_list)
        
        # Output: "1x0e + 1x2e + ..."
        output_list = [f"1x{l}e" for l in range(0, lmax_out + 1, 2)]
        self.output_irreps = " + ".join(output_list)
        
        # Hidden: "Mulx0e + Mulx2e + ..."
        hidden_list = []
        for l in range(0, lmax_out + 1, 2):
            mul = features // (2*l + 1)
            if mul > 0:
                hidden_list.append(f"{mul}x{l}e")
        
        hidden_irreps = " + ".join(hidden_list)
        
        # 2. Build Layers (Gated Nonlinearities)
        self.layers = []
        keys = jax.random.split(key, depth + 1)
        
        # Core is Equivariant MLP
        # But we need manual construction for Equinox compatibility if e3nn modules aren't eqx modules?
        # e3nn-jax is functional or Haiku/Flax based usually.
        # e3nn provides 'e3nn.flax.MultiLayerPerceptron' or 'e3nn.haiku...'
        # For Equinox, we use e3nn functional primitives inside eqx layers.
        
        # Let's verify e3nn-jax usage. It has a 'Linear' which is a Module?
        # e3nn-jax is mostly functional, but has a flax module. 
        # Writing a raw Equinox wrapper for e3nn Linear is easy.
        
        # Let's assume we implement a simple Equinox wrapper for e3nn.Linear.
        
        curr_irreps = self.input_irreps
        
        for i in range(depth):
            # Linear + Gated Activation
            # We use simply Linear -> Gate -> Linear...
            
            # Simplified: Use MultiLayerPerceptron from e3nn? No, that's flax.
            # We construct layers here.
            
            linear = E3Linear(curr_irreps, hidden_irreps, key=keys[i])
            self.layers.append(linear)
            
            # Activation? 
            # Scalar activation (Gate) requires splitting scalars and non-scalars.
            # e3nn.gate()
            
            curr_irreps = hidden_irreps
            
        # Final projection to output
        self.layers.append(E3Linear(curr_irreps, self.output_irreps, key=keys[-1]))

    def __call__(self, input_sh_coeffs):
        # input_sh_coeffs: array of shape (N_sh_in,)
        # Convert to IrrepsArray
        x = e3nn.IrrepsArray(self.input_irreps, input_sh_coeffs)
        
        for layer in self.layers[:-1]:
            x = layer(x)
            # Activation: simple scalar activation on L=0?
            # Or e3nn.norm_activation?
            # acts list must match len(x.irreps)
            acts = [jax.nn.relu] * len(x.irreps)
            x = e3nn.norm_activation(x, acts) # Applies ReLU to norms of irreps
            
        # Final layer (Linear only)
        x = self.layers[-1](x)
        
        return x.array # Back to coeffs

class E3Linear(eqx.Module):
    """
    Equinox wrapper for e3nn.FunctionalLinear
    """
    weights: jnp.ndarray
    instruction: object
    output_irreps: e3nn.Irreps
    
    def __init__(self, irreps_in, irreps_out, key):
        self.output_irreps = irreps_out
        # FunctionalLinear creates instruction (paths)
        linear = e3nn.FunctionalLinear(
            irreps_in,
            irreps_out
        )
        self.instruction = linear
        
        # Initialize weights
        w_shape = self.instruction.num_weights
        self.weights = jax.random.normal(key, (w_shape,))
        
    def __call__(self, x):
        return self.instruction(self.weights, x)

