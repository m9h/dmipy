
import jax.numpy as jnp
from dmipy_jax.algebra.identifiability import construct_polynomial_system, analyze_identifiability, define_signal_components
import logging

def check_identifiability(bvalues, model_name="SphereGPD"):
    """
    Checks algebraic identifiability of a protocol for a given model.
    
    Args:
        bvalues: Array or list of b-values.
        model_name: Name of the model (determines compartments).
        
    Returns:
        dict: Analysis results (basis_length, is_unique, etc.)
    """
    # 1. Determine Compartments based on model
    if model_name == "SphereGPD":
        # Intra + Extra = 2 compartments
        n_compartments = 2
    elif model_name == "CylinderGPD":
        n_compartments = 2
    elif model_name == "Zeppelin":
        n_compartments = 2
    elif model_name == "BallStick":
        n_compartments = 2
    else:
        # Default fallback
        print(f"Warning: Unknown model {model_name}, assuming 2 compartments.")
        n_compartments = 2
        
    # 2. Preprocess B-values
    # Clean up b-values (round to nearest significant integer for algebra)
    # E.g. 998 -> 1000, 1005 -> 1000.
    # Assuming standard shells.
    # If optimization produces 1234.56, we round to nearest 100?
    # Or strict check?
    # Grobner basis needs ratio consistency.
    # Let's assume optimized b-values cluster.
    # For now, we take them as is but converted to float.
    
    # Note: construct_polynomial_system expects list of floats/ints.
    b_vals_list = [float(b) for b in bvalues]
    
    # Dummy signal values (symbolic y_observed is handled inside construct? 
    # No, construct expects values for y.
    # We need to solve for GENERAL identifiability, meaning we treat y as symbolic constants?
    # identifiability.py line 77: zip(..., measured_signals).
    # If we pass symbols for measured_signals, sympy will solve for X in terms of y.
    # Let's check identifiability.py signature.
    
    try:
        # We need to construct symbolic Ys if we want general proof.
        import sympy
        y_syms = [sympy.Symbol(f'y_{i}') for i in range(len(b_vals_list))]
        
        polys, variables = construct_polynomial_system(b_vals_list, y_syms, n_compartments)
        
        result = analyze_identifiability(polys, variables)
        
        # Interpret result
        # If basis is [1], it means inconsistent (0 solutions) -> bad system setup?
        # If basis length > variables, usually 0-dimensional (finite).
        # We want to know exact count.
        # This wrapper just returns the analysis dict for now.
        
        return result
        
    except Exception as e:
        print(f"Algebraic Check Failed: {e}")
        return {"error": str(e)}

def print_identifiability_report(result):
    """Prints a human-readable report."""
    if "error" in result:
        print(f"[Algebra] Check Failed: {result['error']}")
        return

    n_vars = len(result.get('variables', []))
    n_basis = result.get('basis_length', 0)
    degrees = result.get('basis_degrees', [])
    
    print(f"\n[Algebraic Identifiability Report]")
    print(f"  Variables: {n_vars} {result.get('variables')}")
    print(f"  Basis Length: {n_basis}")
    print(f"  Max Degrees: {degrees}")
    
    if n_basis >= n_vars:
        print("  Status: Likely Finite Solutions (Identifiable up to permutation)")
        # Ideally check degree product (Bezout) for max count
    else:
        print("  Status: Potential Infinite Solutions (Under-determined)")
