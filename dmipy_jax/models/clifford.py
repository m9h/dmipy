import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

# Cayley Table for Cl(3,0)
# Basis: 1, e1, e2, e3, e12, e13, e23, e123
# Indices: 0, 1, 2, 3, 4, 5, 6, 7
# We need table [8, 8] -> (sign, result_index)

# Hardcoding multiplication table relative to indices
# Rows: Left operand, Cols: Right operand
# Returns (Index, Sign)
CAYLEY_INDICES = [
    [0, 1, 2, 3, 4, 5, 6, 7], # 1 * ...
    [1, 0, 4, 5, 2, 3, 7, 6], # e1 * ... (e1e1=1, e1e2=e12(4), e1e3=e13(5), e1e12=e2(2), e1e13=e3(3), e1e23=e123(7), e1e123=e23(6))
    [2, 4, 0, 6, 1, 7, 3, 5], # e2 * ...
    [3, 5, 6, 0, 7, 1, 2, 4], # e3 * ...
    [4, 2, 1, 7, 0, 6, 5, 3], # e12 * ...
    [5, 3, 7, 1, 6, 0, 4, 2], # e13 * ...
    [6, 7, 2, 3, 5, 4, 0, 1], # e23 * ...
    [7, 6, 5, 4, 3, 2, 1, 0]  # e123 * ...
]

CAYLEY_SIGNS = [
    [1, 1, 1, 1, 1, 1, 1, 1], # 1
    [1, 1, 1, 1, 1, 1, 1, 1], # e1 (e1e2=e12, e1e12=e2, etc check signs below)
    [1, -1, 1, 1, -1, 1, 1, -1], # e2 (e2e1=-e12(4), etc) - wait let's derive properly
    # A bit complex to hardcode manually without errors.
    # Let's use a simpler formulation:
    # W * X = sum_i sum_j w_i x_j (e_i e_j)
    # We construct the 8x8 matrix M(w) such that M(w) x represents w * x in vector form.
    # Column j of M(w) is w * basis_j.
    # We can precompute this logic.
    [1, 1, 1, 1, 1, 1, 1, 1], # Placeholder, I'll implement logic in __init__?
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

# Better: structure constants tensor S[k, i, j] such that (e_i e_j) = S[k,i,j] e_k.
# Actually S[k, i, j] is sparse (-1, 0, 1).
# We can define it manually.
# i, j -> (k, sign)
# 0 (1): 0,1->1; 0,2->2 ... 0,i->i (sign +)
# 1 (e1): 1,0->1(+); 1,1->0(+1); 1,2->4(+1); 1,3->5(+1); 1,4->2(+1); 1,5->3(+1); 1,6->7(+1); 1,7->6(+1)   (e1e12=e2, e1e13=e3, e1e23=e1e2e3=e123)
# 2 (e2): 2,0->2(+); 2,1->4(-1); 2,2->0(+1); 2,3->6(+1); 2,4->1(-1); 2,5->7(-1); 2,6->3(+1); 2,7->5(-1)   (e2e1=-e12, e2e12=e2e1e2=-e1e2e2=-e1, e2e13=e2e1e3=-e1e2e3=-e123, e2e23=e3, e2e123=e2e1e2e3=e13)
# 3 (e3): 3,0->3(+); 3,1->5(-1); 3,2->6(-1); 3,3->0(+1); 3,4->7(+1); 3,5->1(-1); 3,6->2(-1); 3,7->4(+1)
# 4 (e12): 4,0->4(+); 4,1->2(-1); 4,2->1(+1); 4,3->7(+1); 4,4->0(-1); 4,5->6(-1); 4,6->5(+1); 4,7->3(-1)  (e12e1=-e2, e12e2=e1, e12e3=e123, e12e12=-1, e12e13=e12e1e3=-e1e1e2e3=-e23, e12e23=e12e2e3=e1(-1)e3=-e13)
# 5 (e13): ...
# This is getting tedious.
# Alternative: Load 8x8 matrices for the left-multiplication by each basis element.
pass

def get_left_mult_table():
    # Returns (8, 8) matrix of indices, and (8, 8) matrix of signs
    # Rows i: e_i. Cols j: e_j. Cell: e_i * e_j
    # Index Table
    I = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7], # 1
        [1, 0, 4, 5, 2, 3, 7, 6], # e1
        [2, 4, 0, 6, 1, 7, 3, 5], # e2
        [3, 5, 6, 0, 7, 1, 2, 4], # e3
        [4, 2, 1, 7, 0, 6, 5, 3], # e12
        [5, 3, 7, 1, 6, 0, 4, 2], # e13
        [6, 7, 2, 3, 5, 4, 0, 1], # e23
        [7, 6, 5, 4, 3, 2, 1, 0]  # e123
    ])
    # Sign Table
    # (Checking anti-commutations)
    # 1 commutes with everything (+)
    # e1: e1e2=+e12, e1e3=+e13, e1e12=(+e2), e1e13=(+e3), e1e23=(+e123), e1e123=(+e23) -> Row 1 is all +
    # e2: e2e1=-e12, e2e3=+e23, e2e12=-e1, e2e13=-e123, e2e23=+e3, e2e123=-e13
    # Row 2 signs: +, -, +, +, -, -, +, -
    # e3: e3e1=-e13, e3e2=-e23, e3e12=+e123, e3e13=-e1, e3e23=-e2, e3e123=+e12
    # Row 3 signs: +, -, -, +, +, -, -, +
    # e12: e12e1=-e2, e12e2=+e1, e12e3=+e123, e12e12=-1, e12e13=-e23, e12e23=+e13, e12e123=-e3
    # Row 4 signs: +, -, +, +, -, -, +, -
    # e13: e13e1=-e3, e13e2=-e123, e13e3=+e1, e13e12=+e23, e13e13=-1, e13e23=-e12, e13e123=+e2
    # Row 5 signs: +, -, -, +, +, -, -, +
    # e23: e23e1=+e123, e23e2=-e3, e23e3=+e2, e23e12=-e13, e23e13=+e12, e23e23=-1, e23e123=-e1
    # Row 6 signs: +, +, -, +, -, +, -, -
    # e123: e123e1=+e23, e123e2=-e13, e123e3=+e12, e123e12=-e3, e123e13=+e2, e123e23=-e1, e123e123=-1
    # Row 7 signs: +, +, -, +, -, +, -, -
    
    S = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, -1, 1, 1, -1, -1, 1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, 1, -1, 1, -1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1]
    ])
    return I, S

import numpy as np

class CliffordConv3d(eqx.Module):
    """
    Clifford Convolution (Cl(3,0)).
    """
    # weights: removed
    # We will implement simplified version: Single Channel Clifford Conv.
    # X: (8, D, H, W) -> Y: (8, D, H, W).
    # Weight: (8, K, K, K).
    
    weight: Float[Array, "8 K D H W"]
    bias: Float[Array, "8"]
    
    def __init__(self, key):
        self.weight = jax.random.normal(key, (8, 3, 3, 3)) * 0.1
        self.bias = jnp.zeros(8)
        
    def __call__(self, x):
        # x: (8, D, H, W)
        # w: (8, 3, 3, 3)
        
        # We need to compute Y = W *_g X.
        # Y_k = sum_{i,j} C_{ijk} (W_i * X_j)
        
        # Precompute C_{ijk}: Is 1 if e_i e_j = +e_k, -1 if .., 0 else.
        # From tables:
        # We have I[i, j] giving index k. S[i, j] giving sign.
        # We can form a dense tensor S_idx[k, i, j] which is sign if k=I[i,j] else 0.
        
        I, S = get_left_mult_table()
        # Create sparse-ish tensor M[k, i, j]
        M = np.zeros((8, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                k = I[i, j]
                s = S[i, j]
                M[k, i, j] = s
                
        M_jax = jnp.array(M) # (8_out, 8_w, 8_x)
        
        # Convolve every component of W with every component of X
        # W: (8w, K, K, K)
        # X: (8x, D, H, W)
        # Standard convolution: using grouped conv or looping.
        # Since 8 is small, we can just run 64 convolutions? Or 1 grouped conv?
        # Stack W and X into large channels?
        # Conv input: (1, 8, D, H, W). Kernel: (8*8?, 1, K, K, K)? No.
        
        # Easiest: Only 8x8=64 pairs. Loop/vmap is fine.
        # But jax.lax.conv_general_dilated is fast.
        # Input: (1, 8x, D, H, W)
        # Output: (1, 8w*8x, D, H, W) -> Summing?
        
        # Let's do:
        # Reshape X -> (1, 8, D, H, W)
        # Kernel: We want output of shape (8out, D, H, W).
        # But each output channel 8out is a sum of 64 convs?
        
        # Let's use linearity.
        # Y_k = sum_{i,j} M[k,i,j] conv(x_j, w_i).
        # Let's separate the mixing and the conv.
        # Actually conv(x_j, w_i) depends on i and j.
        # We can run a depthwise conv?
        # Input (8, D, H, W).
        # We want to convolute with W (8, K, K, K).
        # If we treat X as 1 big input channel? No.
        
        # Valid approach:
        # 1. Compute all cross-convolutions between w_i and x_j.
        #    Input: X (1, 8, ...).
        #    Filters: W (8, 1, K, K, K). -> Produces 8 outputs per input channel?
        #    Use `feature_group_count=1`. Input 8. Output 64? (8 filters per input channel).
        #    We construct a kernel of shape (64, 1, K, K, K)? No.
        #    Full transform: Input 8. Output 64.
        #    Kernel shape (64, 8, K, K, K).
        #    We want the (i, j) conv.
        #    Set kernel[i*8+j, j, ...] = W[i, ...]. All others 0.
        #    This is sparse kernel.
        
        # Alternative approach (Feature-wise):
        # Y = M_{kij} (W_i * X_j).
        # Swap sum order: Y_k = sum_i W_i * (sum_j M_{kij} X_j).
        # Let Z_{ki} = sum_j M_{kij} X_j. (Linear mix of input channels).
        # Y_k = sum_i W_i * Z_{ki}.
        # This means for each output blade k, and each weight blade i, we construct a specific input mix Z_{ki}.
        # Then convolve and sum.
        # This requires 64 convs.
        
        # Optimization:
        # Clifford convolution is just a 8-channel -> 8-channel convolution where the 8x8 weight matrix at each spatial location is constrained to be a geometric product matrix.
        # General 8->8 conv kernel has 64 params per voxel.
        # Clifford 8->8 conv kernel W_matrix(w) has 8 params (linear combo).
        # W_matrix = sum_i w_i * BasisMatrix_i.
        # BasisMatrix_i is the matrix for left-mult by e_i.
        
        # So we construct the effective kernel:
        # EffKernel (8out, 8in, K, K, K) = sum_i W[i] * BasisMatrix_i.
        # Then run standard conv!
        
        # BasisMatrices B[i] = M[:, i, :] (from above M tensor).
        # W: (8, K, K, K).
        # Eff: (8out, 8in, K, K, K).
        # Einsum: 'i D H W, o i j -> o j D H W' (Careful with indices)
        # M[k, i, j]: k=out, i=weight, j=input.
        # EffKernel[k, j, ...] = sum_i M[k, i, j] * W[i, ...]
        
        kernel = jnp.einsum('kij, i... -> kj...', M_jax, self.weight)
        
        # Now run standard conv
        # X: (8, D, H, W)
        # Kernel: (8, 8, K, K, K)
        return jax.lax.conv(
            x[None, ...], kernel, (1,1,1), 'SAME'
        )[0] + self.bias[:, None, None, None]

