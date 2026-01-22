
import jax
import jax.numpy as jnp
import equinox as eqx

class FreeWaterNet(eqx.Module):
    """
    MLP for estimating Free Water Fraction from Single-Shell dMRI.
    
    Architecture:
    Input -> MLP(width=256, depth=3) -> Sigmoid -> f_iso
    """
    mlp: eqx.nn.MLP
    
    def __init__(self, in_size, width_size=256, depth=3, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.gelu,
            key=key
        )
        
    def __call__(self, x):
        """
        Args:
            x: (in_size,) signal vector.
        Returns:
            f_iso: scalar [0, 1].
        """
        # eqx.nn.MLP returns (out_size,)
        out = self.mlp(x)
        return jax.nn.sigmoid(out)[0]
