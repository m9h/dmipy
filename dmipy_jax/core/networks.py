import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import List, Callable, Optional

class SineLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    omega_0: float
    is_first: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.Array,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        self.omega_0 = omega_0
        self.is_first = is_first
        
        # Sitzmann et al. initialization
        if is_first:
            bound = 1 / in_features
        else:
            bound = jnp.sqrt(6 / in_features) / omega_0
            
        self.weight = jr.uniform(
            key, (out_features, in_features), minval=-bound, maxval=bound
        )
        self.bias = jr.uniform(
            key, (out_features,), minval=-bound, maxval=bound
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sin(self.omega_0 * (self.weight @ x + self.bias))

class SIREN(eqx.Module):
    layers: List[eqx.Module]
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
        key: jax.Array,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        keys = jr.split(key, hidden_layers + 2)
        
        self.layers = []
        
        # First layer
        self.layers.append(
            SineLayer(
                in_features, 
                hidden_features, 
                key=keys[0], 
                omega_0=first_omega_0, 
                is_first=True
            )
        )
        
        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                SineLayer(
                    hidden_features, 
                    hidden_features, 
                    key=keys[i+1], 
                    omega_0=hidden_omega_0
                )
            )
            
        # Final linear layer
        final_linear = eqx.nn.Linear(hidden_features, out_features, key=keys[-1])
        
        # Initialize final layer weights to be small distributed around 0
        # This helps with convergence
        bound = jnp.sqrt(6 / hidden_features) / hidden_omega_0
        final_linear = eqx.tree_at(
            lambda l: l.weight,
            final_linear,
            jr.uniform(keys[-1], (out_features, hidden_features), minval=-bound, maxval=bound)
        )
        
        self.layers.append(final_linear)
        self.layers = tuple(self.layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x
