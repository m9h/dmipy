import jax
import jax.numpy as jnp

def mps_decomposition(tensor, ranks):
    """
    Decomposes a 4D tensor into a Matrix Product State (Tensor Train).
    
    Args:
        tensor: Input tensor of shape (N1, N2, N3, N4).
        ranks: List of max ranks [r1, r2, r3]. 
               The decomposition will produce cores of shapes:
               G1: (1, N1, r1)
               G2: (r1, N2, r2)
               G3: (r2, N3, r3)
               G4: (r3, N4, 1)
               
    Returns:
        cores: List of 4 tensor cores [G1, G2, G3, G4].
    """
    shape = tensor.shape
    assert len(shape) == 4, "Input tensor must be 4D."
    assert len(ranks) == 3, "Ranks must be a list of 3 integers."
    
    # 1. Reshape to matrix for first SVD: (N1, N2*N3*N4)
    # We want G1: (1, N1, r1)
    
    # Current tensor: T (N1, N2, N3, N4)
    remaining_dim = shape[1] * shape[2] * shape[3]
    mat1 = tensor.reshape(shape[0], remaining_dim)
    
    # SVD
    # U: (N1, min(N1, rem)) -> truncated to r1
    # S: (r1,)
    # V: (r1, rem)
    # G1 = U.reshape(1, N1, r1) (conceptually, or subsume S into next core)
    
    # Using 'full_matrices=False' is crucial.
    u, s, vh = jnp.linalg.svd(mat1, full_matrices=False)
    
    # Truncate to rank r1
    r1 = min(ranks[0], u.shape[1])
    u = u[:, :r1]
    s = s[:r1]
    vh = vh[:r1, :]
    
    # G1 is just U (reshaped). We absorb S into Vh for the next step.
    G1 = u.reshape(1, shape[0], r1)
    
    # Prepare next matrix: S @ Vh -> (r1, N2*N3*N4)
    # Reshape to (r1*N2, N3*N4) for next SVD
    next_mat = (jnp.diag(s) @ vh).reshape(r1 * shape[1], shape[2] * shape[3])
    
    # 2. Second SVD
    u, s, vh = jnp.linalg.svd(next_mat, full_matrices=False)
    
    # Truncate to rank r2
    r2 = min(ranks[1], u.shape[1])
    u = u[:, :r2]
    s = s[:r2]
    vh = vh[:r2, :]
    
    # G2: Reshape U to (r1, N2, r2)
    G2 = u.reshape(r1, shape[1], r2)
    
    # Prepare next matrix
    next_mat = (jnp.diag(s) @ vh).reshape(r2 * shape[2], shape[3])
    
    # 3. Third SVD
    u, s, vh = jnp.linalg.svd(next_mat, full_matrices=False)
    
    # Truncate to rank r3
    r3 = min(ranks[2], u.shape[1])
    u = u[:, :r3]
    s = s[:r3]
    vh = vh[:r3, :]
    
    # G3: Reshape U to (r2, N3, r3)
    G3 = u.reshape(r2, shape[2], r3)
    
    # G4: The remainder is (S @ Vh). It should be (r3, N4) -> reshape to (r3, N4, 1)
    G4 = (jnp.diag(s) @ vh).reshape(r3, shape[3], 1)
    
    return [G1, G2, G3, G4]

def reconstruct_from_mps(cores):
    """
    Reconstructs the full 4D tensor from its MPS cores.
    
    Args:
        cores: List of 4 tensor cores [G1, G2, G3, G4].
               G1: (1, N1, r1)
               G2: (r1, N2, r2)
               G3: (r2, N3, r3)
               G4: (r3, N4, 1)
               
    Returns:
        tensor: Reconstructed 4D tensor (N1, N2, N3, N4).
    """
    G1, G2, G3, G4 = cores
    
    # Contract:
    # G1(1,i,a) * G2(a,j,b) -> T2(1,i,j,b)
    t2 = jnp.tensordot(G1, G2, axes=(2, 0)) # (1, N1, N2, r2)
    
    # T2(1,i,j,b) * G3(b,k,c) -> T3(1,i,j,k,c)
    t3 = jnp.tensordot(t2, G3, axes=(3, 0)) # (1, N1, N2, N3, r3)
    
    # T3(1,i,j,k,c) * G4(c,l,1) -> T4(1,i,j,k,l,1)
    t4 = jnp.tensordot(t3, G4, axes=(4, 0)) # (1, N1, N2, N3, N4, 1)
    
    # Squeeze the auxiliary dimensions
    return jnp.squeeze(t4, axis=(0, 5))

def extract_angular_core(cores):
    """
    Extracts the angular core G4 from the MPS decomposition.
    
    Args:
        cores: List of 4 tensor cores.
        
    Returns:
        angular_core: Tensor of shape (r3, N_ang).
                      This represents the compressed angular information.
    """
    G4 = cores[3]
    # G4 shape is (r3, N_ang, 1). Squeeze last dim.
    return jnp.squeeze(G4, axis=-1)
