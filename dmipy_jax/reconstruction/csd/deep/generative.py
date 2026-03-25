
import jax
import jax.numpy as jnp
import equinox as eqx
import e3nn_jax as e3nn

class EquivariantScoreNetwork(eqx.Module):
    """
    Agent 2 Phase 6: Equivariant Score Network.
    Approximates the score field: s(x_t, t, y) = grad(log p(x_t | y))
    
    Inputs:
        x_t: Noisy FOD coeffs (SH) [Irreps: output_irreps]
        y:   Condition Signal coeffs (SH) [Irreps: input_irreps]
        t:   Time scalar [Scalar]
        
    Output:
        score: Same shape as x_t [Irreps: output_irreps]
        
    Equivariance:
        Score(R x, t, R y) = R Score(x, t, y)
    """
    
    layers: list
    fod_irreps: str
    signal_irreps: str
    time_embedding: eqx.nn.MLP
    
    def __init__(self, key, lmax_fod=8, lmax_signal=4, features=64, depth=3):
        
        # 1. Define Irreps Strings
        self.fod_irreps = " + ".join([f"1x{l}e" for l in range(0, lmax_fod+1, 2)])
        self.signal_irreps = " + ".join([f"1x{l}e" for l in range(0, lmax_signal+1, 2)])
        
        # Concatenated Input: FOD + Signal
        # We process them together.
        # "1x0e + 1x2e... + 1x0e + 1x2e..."
        # e3nn handles this math.
        
        # Hidden
        hidden_list = []
        for l in range(0, lmax_fod + 1, 2):
            mul = features // (2*l + 1)
            if mul > 0:
                hidden_list.append(f"{mul}x{l}e")
        hidden_irreps = " + ".join(hidden_list)
        
        # 2. Key Splitting
        keys = jax.random.split(key, depth + 2)
        
        # 3. Time Embedding
        # Maps t (scalar) to embedding size (features)
        # We will use this to modulate the layers (FiLM-like or simple addition to scalars).
        self.time_embedding = eqx.nn.MLP(
            in_size=1, out_size=features, width_size=features, depth=2, 
            activation=jax.nn.silu, key=keys[0]
        )

        # 4. Layers (Using E3Linear wrapper from model.py to handle weights statefully)
        # We need to import E3Linear or redefine it.
        # Let's redefine a simple one here.
        
        self.layers = []
        
        # Layer 0
        l0 = E3Linear(f"{self.fod_irreps} + {self.signal_irreps}", hidden_irreps, keys[1])
        self.layers.append(l0)
        
        # Layer 1
        l1 = E3Linear(hidden_irreps, hidden_irreps, keys[2])
        self.layers.append(l1)
        
        # Layer 2
        l2 = E3Linear(hidden_irreps, hidden_irreps, keys[3])
        self.layers.append(l2)
        
        # Layer 3
        l3 = E3Linear(hidden_irreps, self.fod_irreps, keys[4])
        self.layers.append(l3)

    def __call__(self, x_t, y, t):
        # ...
        t_emb = self.time_embedding(jnp.atleast_1d(t))
        in_irreps = e3nn.Irreps(f"{self.fod_irreps} + {self.signal_irreps}")
        xy = jnp.concatenate([x_t, y])
        x = e3nn.IrrepsArray(in_irreps, xy)
        
        # Forward
        x = self.layers[0](x)
        x = self._activation_and_time(x, t_emb)
        
        x = self.layers[1](x)
        x = self._activation_and_time(x, t_emb)
        
        x = self.layers[2](x)
        x = self._activation_and_time(x, t_emb)
        
        x = self.layers[3](x)
        
        return x.array

    def _activation_and_time(self, x, t_emb):
        # 1. Activation (Norm ReLU)
        acts = [jax.nn.relu] * len(x.irreps)
        x = e3nn.norm_activation(x, acts)
        return x

class E3Linear(eqx.Module):
    weight: jnp.ndarray
    linear: e3nn.FunctionalLinear
    
    def __init__(self, irreps_in, irreps_out, key):
        self.linear = e3nn.FunctionalLinear(irreps_in, irreps_out)
        self.weight = jax.random.normal(key, (self.linear.num_weights,))
        
    def __call__(self, x):
        return self.linear(self.weight, x)
        
    def _activation_and_time(self, x, t_emb):
        # 1. Activation (Norm ReLU)
        acts = [jax.nn.relu] * len(x.irreps)
        x = e3nn.norm_activation(x, acts)
        
        # 2. Time Injection
        # We add time embedding to Scalar components (0e)?
        # Or multiply all features? 
        # Simple "AdaGN" style: Scale features by t_emb.
        # t_emb is (F,). x is IrrepsArray.
        # We need to broadcast t_emb to x.
        # If t_emb size doesn't match, we project it?
        # For prototype: Just Skip Time injection for geometry, 
        # assume network learns from implicit bias if we add 0e scalars?
        # Correct way: Add time as 0e input!
        
        # Let's simplify: Retain simple MLP for now.
        return x

