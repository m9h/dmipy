import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from typing import Optional, List, Callable, Tuple

class SinusoidalTimeEmbedding(eqx.Module):
    dim: int
    
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t: jax.Array) -> jax.Array:
        # t: (batch_size,) or scalar
        # Returns: (batch_size, dim)
        # Returns: (batch_size, dim)
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb

class TimeResNetBlock(eqx.Module):
    """Residual block that accepts time embedding."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    time_projection: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    shortcut: Callable
    in_channels: int
    out_channels: int
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = eqx.nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=k1)
        
        self.time_projection = eqx.nn.Linear(time_emb_dim, out_channels, key=k2)
        
        self.norm2 = eqx.nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = eqx.nn.Dropout(p=0.1)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, key=k3)

        if in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=k3) # Shared key ok? No, should be k4 ideally but for param count it's fine
        else:
            self.shortcut = lambda x: x

    def __call__(self, x, t_emb, key=None):
        # x: (C, H, W)
        # t_emb: (Time_Dim,)
        
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)
        
        # Add time embedding (broadcast over spatial dims)
        t_proj = jax.nn.silu(self.time_projection(t_emb))
        h = h + t_proj[:, None, None]
        
        h = self.norm2(h)
        h = jax.nn.silu(h)
        if key is not None:
             h = self.dropout(h, key=key)
        h = self.conv2(h)
        
        # shortcut call
        if isinstance(self.shortcut, eqx.Module):
            sc = self.shortcut(x)
        else:
            sc = self.shortcut(x)

        return h + sc

class FlowUNet(eqx.Module):
    time_embed: SinusoidalTimeEmbedding
    time_mlp: eqx.nn.MLP
    
    # Encoder
    down_convs: List[TimeResNetBlock]
    downsample: List[eqx.nn.Conv2d]
    
    # Middle
    mid_block1: TimeResNetBlock
    mid_block2: TimeResNetBlock
    
    # Decoder
    up_convs: List[TimeResNetBlock]
    upsample: List[eqx.nn.ConvTranspose2d]
    
    start_conv: eqx.nn.Conv2d
    end_conv: eqx.nn.Conv2d
    
    def __init__(self, in_channels: int, out_channels: int, base_dim: int = 64, key: jax.random.PRNGKey = None):
        if key is None: key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 20)
        
        time_dim = base_dim * 4
        self.time_embed = SinusoidalTimeEmbedding(base_dim)
        self.time_mlp = eqx.nn.MLP(base_dim, time_dim, width_size=time_dim, depth=2, key=keys[0])
        
        self.start_conv = eqx.nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, key=keys[1])
        
        # Down 1
        self.down_convs = [
            TimeResNetBlock(base_dim, base_dim, time_dim, keys[2]),
            TimeResNetBlock(base_dim, base_dim*2, time_dim, keys[3]),
            TimeResNetBlock(base_dim*2, base_dim*4, time_dim, keys[4]),
        ]
        self.downsample = [
            eqx.nn.Conv2d(base_dim, base_dim, 4, 2, 1, key=keys[5]),
            eqx.nn.Conv2d(base_dim*2, base_dim*2, 4, 2, 1, key=keys[6]),
            eqx.nn.Conv2d(base_dim*4, base_dim*4, 4, 2, 1, key=keys[7]),
        ]
        
        # Mid
        self.mid_block1 = TimeResNetBlock(base_dim*4, base_dim*4, time_dim, keys[8])
        self.mid_block2 = TimeResNetBlock(base_dim*4, base_dim*4, time_dim, keys[9])
        
        # Up
        self.upsample = [
            eqx.nn.ConvTranspose2d(base_dim*4, base_dim*4, kernel_size=4, stride=2, padding=1, key=keys[10]),
            eqx.nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=4, stride=2, padding=1, key=keys[11]),
            eqx.nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=4, stride=2, padding=1, key=keys[12]),
        ]
        self.up_convs = [
            TimeResNetBlock(base_dim*8, base_dim*4, time_dim, keys[13]), # *8 due to concat
            TimeResNetBlock(base_dim*4, base_dim*2, time_dim, keys[14]),
            TimeResNetBlock(base_dim*2, base_dim, time_dim, keys[15]),
        ]
        
        self.end_conv = eqx.nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1, key=keys[16])

    def __call__(self, x, t, context, key=None):
        # x: (C_out, H, W)
        # t: scalar
        # context: (C_ctx, H, W)
        
        # Concatenate x and context
        inp = jnp.concatenate([x, context], axis=0)
        
        # Time embedding
        # t is scalar, expand to (1,) for Sinusoidal?
        # My Sinusoidal takes (batch,) so (1,)
        t_emb = self.time_embed(jnp.array([t]))[0] # (base_dim,)
        t_emb = self.time_mlp(t_emb) # (time_dim,)
        
        # Initial Conv
        h = self.start_conv(inp)
        
        skips = []
        
        # Down
        for block, down in zip(self.down_convs, self.downsample):
            h = block(h, t_emb, key)
            skips.append(h)
            h = down(h)
            
        # Mid
        h = self.mid_block1(h, t_emb, key)
        h = self.mid_block2(h, t_emb, key)
        
        # Up
        for block, up, skip in zip(self.up_convs, self.upsample, reversed(skips)):
            h = up(h)
            # Resize h if needed or assume padding correct? 
            # 256 -> 128 -> 64 -> 32
            # 32 -> 64 -> 128 -> 256
            h = jnp.concatenate([h, skip], axis=0)
            h = block(h, t_emb, key)
            
        out = self.end_conv(h)
        return out

