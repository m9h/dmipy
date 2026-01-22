import jax
import jax.numpy as jnp
import equinox as eqx
import healpy as hp
import numpy as np
from typing import Optional, List
from jaxtyping import Array, Float, Complex

def get_healpix_laplacian(nside: int, nest: bool = True) -> Float[Array, "metrics metrics"]:
    """
    Computes the combinatorial Laplacian for a Healpix grid.
    Returns L = I - D^(-1/2) A D^(-1/2).
    """
    npix = hp.nside2npix(nside)
    # Get neighbors (8 neighbors for most, 7 for some)
    # hp.get_all_neighbours returns (8, npix). -1 for missing neighbors.
    neighbors = hp.get_all_neighbours(nside, np.arange(npix), nest=nest)
    
    # Build Adjacency Matrix
    # We use numpy for construction as this involves boolean logic/indexing not great for JAX trace
    # but we return a JAX array.
    
    # Efficient sparse construction would be better for memory, 
    # but for typical dMRI (nside=16, npix=3072) dense is fine (3k x 3k = 9M float32 = 36MB).
    
    A = np.zeros((npix, npix), dtype=np.float32)
    
    for i in range(npix):
        for n in neighbors[:, i]:
            if n != -1:
                A[i, n] = 1.0
                A[n, i] = 1.0 # Symmetry
                
    # Degree matrix
    deg = np.sum(A, axis=1)
    
    # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[np.isinf(deg_sqrt_inv)] = 0.0
    
    D_inv_sqrt = np.diag(deg_sqrt_inv)
    
    L = np.eye(npix) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Rescale to [-1, 1] for Chebyshev: L_tilde = L - I (since lambda_max approx 2)
    L_tilde = L - np.eye(npix)
    
    return jnp.array(L_tilde)

class ChebConv(eqx.Module):
    """
    Chebyshev Graph Convolution on the Sphere.
    Approximates spectral graph convolution using Chebyshev polynomials.
    """
    weights: Float[Array, "K in_channels out_channels"]
    bias: Optional[Float[Array, "out_channels"]]
    K: int = eqx.field(static=True)
    laplacian: Float[Array, "nodes nodes"] # Graph structure
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        nside: int,
        key: jax.Array
    ):
        self.K = K
        w_key, b_key = jax.random.split(key)
        lim = 1 / jnp.sqrt(in_channels * K)
        self.weights = jax.random.uniform(w_key, (K, in_channels, out_channels), minval=-lim, maxval=lim)
        self.bias = jnp.zeros(out_channels)
        self.laplacian = get_healpix_laplacian(nside)
        
    def __call__(self, x: Float[Array, "in_channels nodes"]) -> Float[Array, "out_channels nodes"]:
        """
        Forward pass.
        x: (C_in, V)
        """
        # Chebyshev recurrence
        # T_0(x) = x
        # T_1(x) = L x
        # T_k(x) = 2 L T_{k-1}(x) - T_{k-2}(x)
        
        # x shape: (Cin, V)
        # We need to perform matmuls over V.
        # Let's verify dimensions.
        # L: (V, V)
        # x.T: (V, Cin)
        # L @ x.T -> (V, Cin) -> transpose -> (Cin, V)
        
        L = self.laplacian
        
        # Tx list stores [T_0 @ x, T_1 @ x, ...]
        # Each element is (Cin, V)
        
        Tx_0 = x
        Tx_1 = (L @ x.T).T # (V, V) @ (V, C) -> (V, C) -> (C, V)
        
        Tx = [Tx_0, Tx_1]
        
        for k in range(2, self.K):
            Tx_k = 2 * (L @ Tx[-1].T).T - Tx[-2]
            Tx.append(Tx_k)
            
        # Convolve: sum_k (Tx_k @ W_k)
        # Tx: list of K arrays of shape (Cin, V)
        # Stack -> (K, Cin, V)
        Tx_stack = jnp.stack(Tx, axis=0)
        
        # Weights: (K, Cin, Cout)
        # Output: (Cout, V)
        # Contract over K and Cin
        
        # Einsum: k c_i v, k c_i c_o -> c_o v
        out = jnp.einsum('kiv,kio->ov', Tx_stack, self.weights)
        
        if self.bias is not None:
            # Bias is (Cout,) -> broadcast to (Cout, V)
            out = out + self.bias[:, None]
            
        return out


class IsoConv3D(eqx.Module):
    """
    Isotropic 3D Convolution.
    Enforces rotational invariance at the local level by sharing weights
    among equidistant voxel neighbors (Center, Face, Edge, Corner).
    kernel_size is fixed to 3.
    """
    params: Float[Array, "4 in_channels out_channels"] # 4 groups of weights
    bias: Optional[Float[Array, "out_channels"]]
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        key: jax.Array
    ):
        w_key, b_key = jax.random.split(key)
        # 4 parameters: Center, Face, Edge, Corner
        lim = 1 / jnp.sqrt(in_channels * 27)
        self.params = jax.random.uniform(w_key, (4, in_channels, out_channels), minval=-lim, maxval=lim)
        self.bias = jnp.zeros(out_channels)
        
    def __call__(self, x: Float[Array, "in_channels D H W"]) -> Float[Array, "out_channels D H W"]:
        # Construct the 3x3x3 kernel from the 4 parameters
        # Indices:
        # Center: (1,1,1) -> dist 0
        # Faces: 6 neighbors -> dist 1
        # Edges: 12 neighbors -> dist sqrt(2)
        # Corners: 8 neighbors -> dist sqrt(3)
        
        # We can build a template or index map.
        # Kernel shape: (C_out, C_in, 3, 3, 3) for jax.lax.conv
        
        k_center = self.params[0] # (Cin, Cout)
        k_face = self.params[1]
        k_edge = self.params[2]
        k_corner = self.params[3]
        
        # Build 3x3x3 layout
        # (3,3,3)
        layout = np.zeros((3,3,3), dtype=np.int32)
        # Center
        layout[1,1,1] = 0
        # Faces
        faces = [(0,1,1), (2,1,1), (1,0,1), (1,2,1), (1,1,0), (1,1,2)]
        for idx in faces: layout[idx] = 1
        # Edges (remaining with one 1) -> No, edges have two non-1s?
        # Let's count coordinates != 1.
        # Center: #!=1 is 0.
        # Face: #!=1 is 1. (e.g. 0,1,1)
        # Edge: #!=1 is 2. (e.g. 0,0,1)
        # Corner: #!=1 is 3. (e.g. 0,0,0)
        
        # We can generate this map dynamically or hardcode.
        coords = np.indices((3,3,3)).reshape(3, -1).T # (27, 3)
        dist_sq = np.sum((coords - 1)**2, axis=1) # 0, 1, 2, 3
        
        # Construct kernel (Cin, Cout, 3, 3, 3) -> then transpose to (Cout, Cin, 3, 3, 3) for conv_general
        # Actually standard weights are usually (Out, In, Spatial...)
        
        kernel_flat = []
        for d in dist_sq:
            if d == 0: param = k_center
            elif d == 1: param = k_face
            elif d == 2: param = k_edge
            elif d == 3: param = k_corner
            kernel_flat.append(param)
            
        # Stack: (27, Cin, Cout)
        kernel_flat = jnp.stack(kernel_flat, axis=0)
        
        # Reshape to (3, 3, 3, Cin, Cout)
        kernel = kernel_flat.reshape(3, 3, 3, x.shape[0], -1)
        
        # Permute to (Cout, Cin, 3, 3, 3)
        kernel = jnp.permute_dims(kernel, (4, 3, 0, 1, 2))
        
        # Run convolution
        return jax.lax.conv(
            x[None, ...], # Add batch dim (1, C, D, H, W)
            kernel, # (Cout, Cin, 3, 3, 3)
            window_strides=(1,1,1),
            padding='SAME'
        )[0] + self.bias[:, None, None, None]

