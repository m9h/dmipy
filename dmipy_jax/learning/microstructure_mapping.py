
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Callable

class MicrostructureMapper(eqx.Module):
    """
    A simple MLP to learn the mapping between Histology features and dMRI signals (or parameters).
    
    Can be used for:
    1. Forward Mapping: Histology (e.g., radius, density) -> Signal
    2. Inverse Mapping: Signal -> Histology
    """
    layers: List[eqx.nn.Linear]
    activation: Callable
    
    def __init__(self, 
                 in_size: int, 
                 out_size: int, 
                 width_size: int = 64, 
                 depth: int = 3, 
                 activation: Callable = jax.nn.relu,
                 key: Optional[jax.random.PRNGKey] = None):
        """
        Args:
            in_size: Dimension of input (e.g., 2 for radius+density).
            out_size: Dimension of output (e.g., N_bvals for signal).
            width_size: Width of hidden layers.
            depth: Number of hidden layers.
            activation: Activation function.
            key: PRNGKey for initialization.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
            
        keys = jax.random.split(key, depth + 1)
        
        self.layers = []
        
        # Input layer
        self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
        
        # Hidden layers
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i+1]))
            
        # Output layer
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[depth]))
        
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input array of shape (in_size,).
        Returns:
            Output array of shape (out_size,).
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Linear output
        x = self.layers[-1](x)
        return x
