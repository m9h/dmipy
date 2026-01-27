import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, List, Callable

class DownSample(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.BatchNorm
    apply_batchnorm: bool

    def __init__(self, in_channels: int, out_channels: int, size: int = 4, apply_batchnorm: bool = True, key: Optional[jax.random.PRNGKey] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, use_bias=False, key=key)
        self.apply_batchnorm = apply_batchnorm
        if apply_batchnorm:
            self.norm = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        else:
            self.norm = None

    def __call__(self, x, state: eqx.nn.State, inference: bool = False, key: Optional[jax.random.PRNGKey] = None):
        h = self.conv(x)
        if self.apply_batchnorm:
            h, state = self.norm(h, state, inference=inference)
        h = jax.nn.leaky_relu(h, negative_slope=0.2)
        return h, state

class UpSample(eqx.Module):
    conv: eqx.nn.ConvTranspose2d
    norm: eqx.nn.BatchNorm
    dropout: eqx.nn.Dropout
    apply_dropout: bool

    def __init__(self, in_channels: int, out_channels: int, size: int = 4, apply_dropout: bool = False, key: Optional[jax.random.PRNGKey] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.conv = eqx.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, use_bias=False, key=key)
        self.norm = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        self.apply_dropout = apply_dropout
        self.dropout = eqx.nn.Dropout(p=0.5)

    def __call__(self, x, skip_input, state: eqx.nn.State, inference: bool = False, key: Optional[jax.random.PRNGKey] = None):
        h = self.conv(x)
        h, state = self.norm(h, state, inference=inference)
        if self.apply_dropout:
            h = self.dropout(h, key=key, inference=inference)
        h = jax.nn.relu(h)
        # Concatenate with skip connection
        # Check shapes if necessary, but assuming padding='same' works out
        h = jnp.concatenate([h, skip_input], axis=0) # Concatenate along channel axis (0 in CHW)
        return h, state

class DeepFCDGenerator(eqx.Module):
    down_stack: List[DownSample]
    up_stack: List[UpSample]
    last: eqx.nn.ConvTranspose2d
    
    def __init__(self, input_channels: int = 7, output_channels: int = 1, key: Optional[jax.random.PRNGKey] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 16)
        
        # Down stack
        self.down_stack = [
            DownSample(input_channels, 64, apply_batchnorm=False, key=keys[0]),
            DownSample(64, 128, key=keys[1]),
            DownSample(128, 256, key=keys[2]),
            DownSample(256, 512, key=keys[3]),
            DownSample(512, 512, key=keys[4]),
            DownSample(512, 512, key=keys[5]),
            DownSample(512, 512, key=keys[6]),
            DownSample(512, 512, key=keys[7]),
        ]
        
        # Up stack
        self.up_stack = [
            UpSample(512, 512, apply_dropout=True, key=keys[8]),
            UpSample(1024, 512, apply_dropout=True, key=keys[9]), # 1024 because of concat (512+512)
            UpSample(1024, 512, apply_dropout=True, key=keys[10]),
            UpSample(1024, 512, key=keys[11]),
            UpSample(1024, 256, key=keys[12]),
            UpSample(512, 128, key=keys[13]),
            UpSample(256, 64, key=keys[14]),
        ]
        
        self.last = eqx.nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1, key=keys[15])

    def __call__(self, x, state: eqx.nn.State, inference: bool = False, key: Optional[jax.random.PRNGKey] = None):
        # x shape: (C, H, W)
        skips = []
        h = x
        
        # Downsampling
        for down in self.down_stack:
            h, state = down(h, state, inference=inference, key=key)
            skips.append(h)
        
        skips = reversed(skips[:-1])
        
        # Upsampling
        for up, skip in zip(self.up_stack, skips):
            # Split key if dropout is used
            if up.apply_dropout and key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
                
            h, state = up(h, skip, state, inference=inference, key=subkey)
            
        h = self.last(h)
        h = jnp.tanh(h)
        
        return h, state

class PatchGANDiscriminator(eqx.Module):
    down1: DownSample
    down2: DownSample
    down3: DownSample
    conv4: eqx.nn.Conv2d
    norm4: eqx.nn.BatchNorm
    last: eqx.nn.Conv2d
    
    def __init__(self, input_channels: int = 8, key: Optional[jax.random.PRNGKey] = None):
        # Input channels: 7 (input stack) + 1 (target) = 8
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 5)
        
        self.down1 = DownSample(input_channels, 64, apply_batchnorm=False, key=keys[0])
        self.down2 = DownSample(64, 128, key=keys[1])
        self.down3 = DownSample(128, 256, key=keys[2])
        
        # "ZeroPad1" + Conv 512 (stride 1) + Batchnorm + LeakyRelu in TF impl
        # TF Padding: 'same' corresponds to size // 2 roughly, but stride 1 with padding?
        # TF Code: ZeroPadding2D makes it larger, then valid conv? Or Same?
        # TF Code: ZeroPadding2D()(down3) -> Conv2D(512, 4, strides=1, kernel_intializer, use_bias=False)
        # This implies explicit padding to handle boundaries for "PatchGAN" effect.
        # We'll use padding=1 for 4x4 kernel? 
        self.conv4 = eqx.nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, use_bias=False, key=keys[3])
        self.norm4 = eqx.nn.BatchNorm(512, axis_name="batch")
        
        self.last = eqx.nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, key=keys[4])

    def __call__(self, inp, tar, state: eqx.nn.State, inference: bool = False, key: Optional[jax.random.PRNGKey] = None):
        # Concatenate input and target along channel dimension
        x = jnp.concatenate([inp, tar], axis=0)
        
        h, state = self.down1(x, state, inference=inference)
        h, state = self.down2(h, state, inference=inference)
        h, state = self.down3(h, state, inference=inference)
        
        h = self.conv4(h)
        h, state = self.norm4(h, state, inference=inference)
        h = jax.nn.leaky_relu(h, negative_slope=0.2)
        
        h = self.last(h)
        return h, state
