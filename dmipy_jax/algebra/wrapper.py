
import jax
import jax.numpy as jnp
from dmipy_jax.algebra.identifiability import construct_polynomial_system, analyze_identifiability, define_signal_components, compute_invariants
import logging
import sympy

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

def get_model_invariants(b_values, model_name="SphereGPD"):
    """
    Returns JAX-compilable functions P(signal) that must be zero on the model manifold.
    
    Returns:
        invariant_fn(signal): JAX function returning array of residuals.
    """
    # 1. Get compartments
    if model_name in ["SphereGPD", "Zeppelin", "BallStick", "CylinderGPD"]:
        n_compartments = 2
    else:
        n_compartments = 2
        
    b_vals_list = [float(b) for b in b_values]
    
    # 2. Compute invariants symbolically
    invariants_sym, y_syms = compute_invariants(b_vals_list, n_compartments)
    
    if not invariants_sym:
        print("No invariants found (or system is trivial/underdetermined).")
        return None
        
    print(f"Found {len(invariants_sym)} invariant polynomials.")
    
    # 3. JAX compilation
    # We want a function fn(y_array) -> [res1, res2...]
    # Lambdify usually takes *args.
    # invariants_sym is a list of expressions.
    
    # Create a vector function [P1, P2...]
    # lambdify(y_syms, invariants_sym, 'jax')
    
    try:
        jax_func_raw = sympy.lambdify(y_syms, invariants_sym, modules='jax')
    except Exception as e:
        print(f"Lambdify failed: {e}")
        return None
        
    def invariant_fn(signal):
        # signal is JAX array (N_b,)
        # unpack to args
        # Ensure list
        return jnp.stack(jax_func_raw(*signal))
        
    return invariant_fn
