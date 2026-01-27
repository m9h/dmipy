import jax
import jax.numpy as jnp
import ttax
from dmipy_jax.invariants import compute_invariants_jax

def tt_decompose_signal(signal_volume, rank=10):
    """
    Perform TT-decomposition on a dMRI signal volume.
    
    Args:
        signal_volume: (B, X, Y, Z, Dir) array of dMRI signals.
                        Note: TT decomposition works on the whole tensor.
                        If B is large, might be better to decompose per-subject or just (X,Y,Z,Dir).
                        Assuming input is a single 5D block.
        rank: TT-rank to use for decomposition or truncation. 
              Can be an integer (constant rank) or list of ranks.
              
    Returns:
        tt_tensor: A ttax.TTTensor object representing the compressed signal.
    """
    # Manual TT-SVD implementation compatible with JAX
    # Reshape signal_volume to flat tensor if not already
    # signal_volume shape: (B, X, Y, Z, Dir)
    # TT dimensions: we treat it as d-dimensional tensor.
    # Dimensions: n1=B, n2=X, n3=Y, n4=Z, n5=Dir.
    shape = signal_volume.shape
    d = len(shape)
    
    # Handle rank argument
    if isinstance(rank, int):
        max_ranks = [rank] * (d - 1)
    else:
        max_ranks = rank
        
    cores = []
    
    # Initial matrix
    temp = signal_volume
    # Dimensionality tracking
    curr_rank = 1
    
    for k in range(d - 1):
        n_k = shape[k]
        
        # Current flattening: (curr_rank * n_k, -1)
        temp_flat = temp.reshape(curr_rank * n_k, -1)
        
        # SVD
        # full_matrices=False is standard
        u, s, vt = jnp.linalg.svd(temp_flat, full_matrices=False)
        
        # Truncate
        # Determine new rank
        r_next = min(len(s), max_ranks[k] if k < len(max_ranks) else max_ranks[-1])
        
        u = u[:, :r_next]
        s = s[:r_next]
        vt = vt[:r_next, :]
        
        # u is (curr_rank * n_k, r_next). Reshape to core (curr_rank, n_k, r_next)
        core = u.reshape(curr_rank, n_k, r_next)
        cores.append(core)
        
        # Prepare next temp
        # W = S @ Vt -> (r_next, remaining)
        temp = jnp.diag(s) @ vt
        curr_rank = r_next
        
    # After loop, temp is (r_{d-1}, n_d). This is the last core.
    # Reshape to (r_{d-1}, n_d, 1)
    last_core = temp.reshape(curr_rank, shape[-1], 1)
    cores.append(last_core)
    
    # Construct ttax tensor
    tt_compressed = ttax.TT(cores)
    
    return tt_compressed


def global_local_bridge(tt_tensor, angular_dim_index=-1):
    """
    Apply compute_invariants_jax to the angular core of the TT-decomposition,
    then project the result back to spatial coordinates using the spatial cores.
    
    Args:
        tt_tensor: ttax.TTTensor object (5D: B, X, Y, Z, Dir).
        angular_dim_index: Index of the angular dimension (default last).
        
    Returns:
        spatial_invariants: (B, X, Y, Z, N_invariants) array of reconstructed invariants.
    """
    # Get all cores
    cores = list(tt_tensor.tt_cores)
    
    # Identify angular core
    # ttax cores are (r_i, n_i, r_{i+1})
    ang_core = cores[angular_dim_index] # Shape (r_in, Dir, r_out)
    
    # We want to treat the angular core as a collection of "signals" in the Dir dimension
    # and compute invariants for each (r_in, r_out) pair.
    # Reshape: (r_in * r_out, Dir) -> coeffs?
    # No, invariants computation expects (Batch, Dir/Coeffs).
    # The 'Dir' dimension here is likely the Angular dimension, which implies it 
    # contains the Signal values on the sphere (if raw signal) 
    # OR SH coefficients (if the input was SH coeffs).
    # The spec say "takes Spherical Harmonic coefficients" for compute_invariants_jax.
    # But `tt_decompose_signal` takes `signal_volume`. 
    # If the input volume is raw signal (e.g. b-shells), we first need to map 
    # the angular core to SH coefficients? 
    # Or assume the volume was already SH coeffs? 
    # "takes a (B, X, Y, Z, Dir) dMRI volume". "Dir" usually implies q-space samples.
    # `compute_invariants_jax` takes SH coeffs.
    # WE MISS A STEP: Signal -> SH.
    # However, for the "Bridge", if we apply `compute_invariants_jax` to the angular core,
    # we tacitly assume the angular core represents SH coeffs or can be treated as such.
    # If the angular dimension IS size N_SH, then it's fine.
    # If it is N_gradients, we strictly need a transformation S -> C (SH basis).
    # But the prompt says: "applies compute_invariants_jax to the angular core".
    # This implies the angular core IS the input to `compute_invariants_jax` (appropriately reshaped).
    # Let's assume the angular dimension size matches what `compute_invariants_jax` expects (SH counts).
    # If not, the user should have projected to SH before TT, or we assume implicit mapping.
    
    r_in, n_ang, r_out = ang_core.shape
    
    # Reshape to (Batch, N_coeffs)
    # Batch here is effectively (r_in, r_out) pairs.
    ang_flat = jnp.transpose(ang_core, (0, 2, 1)) # (r_in, r_out, n_ang)
    ang_flat = ang_flat.reshape(-1, n_ang) # (r_in*r_out, n_ang)
    
    # Compute invariants
    # Output: (r_in*r_out, N_inv)
    invariants_flat = compute_invariants_jax(ang_flat)
    n_inv = invariants_flat.shape[-1]
    
    # Reshape back to core-like structure: (r_in, r_out, N_inv)
    # But wait, original core was (r_in, n_ang, r_out).
    # We want to replace the angular core with the invariant core?
    # The "invariants" are new "features". 
    # The spatial reconstruction should result in (B, X, Y, Z, N_inv).
    # The original TT structure reconstructed (B, X, Y, Z, n_ang).
    # If we replace the angular core with something of shape (r_in, N_inv, r_out),
    # and then perform full contraction, we get (B, X, Y, Z, N_inv).
    
    # Reshape invariants to (r_in, r_out, N_inv) -> Transpose to (r_in, N_inv, r_out)
    inv_core = invariants_flat.reshape(r_in, r_out, n_inv)
    inv_core = jnp.transpose(inv_core, (0, 2, 1)) # (r_in, N_inv, r_out)
    
    # Replace the angular core in the list
    new_cores = list(cores)
    new_cores[angular_dim_index] = inv_core
    
    # Reconstruct the tensor
    # But ttax.TTTensor expects specific ranks. The 3rd dim of core[i] must match 1st dim of core[i+1].
    # We didn't change r_in or r_out, only the middle dimension (n_ang -> n_inv). 
    # This is valid for TT structure contraction.
    
    new_tt = ttax.TT(new_cores)
    
    # Full reconstruction to spatial grid
    # ttax.full(new_tt) will result in shape (B, X, Y, Z, N_inv)
    spatial_invariants = ttax.full(new_tt)
    
    return spatial_invariants
