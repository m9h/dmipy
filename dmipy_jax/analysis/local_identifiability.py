import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Tuple

def check_model_identifiability(
    model_func: Any, 
    params: Any, 
    acquisition: Any, 
    parameter_names: List[str] = None
) -> Dict[str, Any]:
    """
    Checks Local Structural Identifiability using the Jacobian Rank condition.
    
    Args:
        model_func: Callable(params, acquisition) -> signal
        params: Flat parameter vector at which to evaluate identifiability.
        acquisition: Acquisition object passed to model.
        parameter_names: List of parameter names for reporting.
        
    Returns:
        Dictionary containing:
        - singular_values: SV of sensitivity matrix.
        - rank: Numeric rank of Jacobian.
        - null_space: V matrix columns corresponding to zero singular values.
        - collinear_sets: List of parameters that are collinear.
        - condition_number: Ratio of largest to smallest singular value.
    """
    
    # 1. Compute Jacobian (Sensitivity Matrix)
    # Shape: (N_measurements, N_params)
    J = jax.jacfwd(lambda p: model_func(p, acquisition))(params)
    
    # 2. Check for NaNs
    if jnp.isnan(J).any():
        return {"error": "Jacobian contains NaNs", "jacobian": J}
        
    # 3. Singular Value Decomposition
    # J = U * S * Vh
    # singular_values are S
    # Vh rows (or V columns) correspond to parameter space directions
    U, S, Vh = jnp.linalg.svd(J, full_matrices=False)
    V = Vh.T
    
    # 4. Rank Determination
    # Threshold usually machine epsilon * max(S) * max(shape)
    # We use a conservative threshold
    max_sv = S[0]
    threshold = 1e-5 * max_sv # Relative threshold
    
    rank = jnp.sum(S > threshold)
    n_params = len(params)
    
    is_identifiable = (rank == n_params)
    
    # 5. Null Space Analysis
    # If rank < n_params, look at V columns where S is small
    null_indices = jnp.where(S <= threshold)[0]
    
    collinear_params = []
    
    if len(null_indices) > 0:
        if parameter_names is None:
            parameter_names = [f"p{i}" for i in range(n_params)]
            
        for idx in null_indices:
            # Get the null vector
            null_vec = V[:, idx]
            
            # Identify which parameters contribute significantly to this null vector
            # i.e., linear combination of these parameters is unobservable (S ~ 0)
            contributing_indices = jnp.where(jnp.abs(null_vec) > 0.1)[0]
            
            names = [parameter_names[i] for i in contributing_indices]
            coefficients = [float(null_vec[i]) for i in contributing_indices]
            
            collinear_params.append({
                "singular_value": float(S[idx]),
                "params": names,
                "coefficients": coefficients
            })
            
    return {
        "is_identifiable": bool(is_identifiable),
        "rank": int(rank),
        "n_params": int(n_params),
        "condition_number": float(S[0] / (S[-1] + 1e-15)),
        "singular_values": np.array(S),
        "collinear_sets": collinear_params
    }
