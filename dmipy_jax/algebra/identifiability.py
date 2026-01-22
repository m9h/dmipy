
import sympy
from typing import List, Tuple, Dict, Optional

def define_signal_components(n_compartments: int):
    """
    Define symbolic variables for a multi-compartment model.
    Returns:
        variables: List of sympy symbols [f1, D1, f2, D2, ...]
        S0: S0 symbol
    """
    f = [sympy.Symbol(f'f_{i}', real=True, positive=True) for i in range(1, n_compartments + 1)]
    D = [sympy.Symbol(f'D_{i}', real=True, positive=True) for i in range(1, n_compartments + 1)]
    S0 = sympy.Symbol('S0', real=True, positive=True)
    
    # Pack variables
    variables = [S0]
    for i in range(n_compartments):
        variables.append(f[i])
        variables.append(D[i])
        
    return variables, f, D, S0

def construct_polynomial_system(
    b_values: List[float], 
    measured_signals: List[float], # Symbolic or numeric
    n_compartments: int
):
    """
    Constructs a polynomial system from the exponential decay model.
    
    Model: S(b) = S0 * sum(f_i * exp(-b * D_i))
    Transformation: Let x_i = exp(-D_i * scale). This is hard for arbitrary b.
    
    Alternative:
    We treat the system algebraically by assuming b-values are integer multiples 
    of a base b0 (or rational).
    Let X_i = exp(-b_gcd * D_i).
    Then exp(-k * b_gcd * D_i) = X_i^k.
    
    This converts the exponential system into a polynomial system in variables X_i.
    
    Args:
        b_values: List of b-values.
        measured_signals: Observed signals (usually symbolic 'y_j' to keep analysis general).
        n_compartments: Number of compartments.
        
    Returns:
        polynomials: List of sympy polynomials.
        variables: List of symbols [S0, f_i, X_i].
    """
    
    # 1. Find GCD of b-values to define base unit
    # (Assuming integers for simplicity of Grobner basis, or reasonable rationals)
    # If floats, we approximate or assume user provides structured b-values.
    # For rigorous proof, we usually assume b = [0, b0, 2b0, ...].
    
    # Let's assume b_values are multiples of the smallest non-zero b.
    non_zero_b = [b for b in b_values if b > 1e-6]
    if not non_zero_b:
        raise ValueError("Need non-zero b-values.")
        
    b_base = min(non_zero_b)
    
    # Integer multipliers
    ks = [int(round(b / b_base)) for b in b_values]
    
    # Variables
    # We solve for X_i = exp(-b_base * D_i)
    # And w_i = S0 * f_i (weighted fractions)
    
    X = [sympy.Symbol(f'X_{i}') for i in range(1, n_compartments + 1)]
    w = [sympy.Symbol(f'w_{i}') for i in range(1, n_compartments + 1)]
    
    system_polys = []
    
    for k, b_val, y_observed in zip(ks, b_values, measured_signals):
        # Model: S_k = sum(w_i * X_i^k)
        # Poly: sum(w_i * X_i^k) - y_observed = 0
        
        model_term = sum(w[i] * (X[i]**k) for i in range(n_compartments))
        poly = model_term - y_observed
        system_polys.append(poly)
        
    # Constraint: If we want to enforce sum(f) = 1, then sum(w) = S0.
    # But usually we just solve for w_i directly (proton density weighted).
    # If S0 is separate, we add S0 variable. 
    # Let's stick to w_i (S0 * f_i) variables for simplicity of the ideal.
    
    variables = w + X
    return system_polys, variables

def analyze_identifiability(polynomials, variables):
    """
    Computes Grobner Basis to check identifiability.
    
    Args:
        polynomials: List of sympy expressions (P = 0).
        variables: List of variables to solve for.
        
    Returns:
        Dict with analysis results.
    """
    # Compute Grobner Basis
    # Use 'lex' order for elimination, or 'grevlex' for speed.
    # 'lex' is better for seeing the triangular structure.
    
    gb = sympy.groebner(polynomials, variables, order='lex')
    
    # Check dimension
    # (Simplified check: if GB is {1}, assumed contradiction/no solution? 
    # If finite solutions, unique triangular form.)
    
    is_zero_dimensional = False
    
    # Heuristic for zero-dimensional ideal (finite solutions):
    # For every variable v, there is a polynomial in GB with leading term v^k.
    # Or simply: len(GB) >= len(variables) usually (but not strictly).
    
    # SymPy doesn't have a direct 'ideal_dimension' function easily exposed in basic API,
    # but we can check if the system is solvable.
    
    # Let's return the basis length and the basis itself (string form)
    
    # Calculate max possible degree for each polynomial (total degree)
    basis_degrees = []
    for p in gb:
        # Convert to Poly to interact robustly
        poly_obj = sympy.Poly(p, *variables)
        basis_degrees.append(poly_obj.total_degree())

    return {
        "basis_length": len(gb),
        "is_trivial": list(gb) == [1],
        "basis_degrees": basis_degrees,
        "variables": [str(v) for v in variables],
        "basis": [str(p) for p in gb]
    }

def compute_invariants(
    b_values: List[float],
    n_compartments: int
) -> Tuple[List[sympy.Expr], List[sympy.Symbol]]:
    """
    Computes the algebraic invariants of the signal manifold.
    Eliminates model parameters to find relationships strictly between signal measurements.
    
    Args:
        b_values: List of b-values.
        n_compartments: Number of compartments.
        
    Returns:
        invariants: List of sympy polynomials P(y_0, y_1, ...) that should be zero.
        y_syms: The symbols y_0, ... used in the polynomials.
    """
    import sympy
    
    # 1. Setup Symbols
    y_syms = [sympy.Symbol(f'y_{i}') for i in range(len(b_values))]
    
    # 2. Construct System (S(theta) - y = 0)
    system_polys, params = construct_polynomial_system(b_values, y_syms, n_compartments)
    
    # 3. Define Ring Variables: Parameters + Signals
    # We want to eliminate Parameters. Best order: lex(Parameters > Signals)
    # This means params are "more expensive", so GB will try to express them in terms of y,
    # or find equations without them.
    
    all_vars = params + y_syms
    
    # 4. Compute Grobner Basis
    # Note: efficient calculation for elimination is usually Block order (lex block for elimination vars, grevlex for rest).
    # SymPy doesn't support block orders explicitly in groebner() API easily?
    # It does: order='lex' usually works for elimination if vars are ordered correctly.
    # We pass variables in order elimination_first -> keep_last.
    
    print(f"[Identifiability] Computing Grobner Basis to eliminate {len(params)} parameters...")
    gb = sympy.groebner(system_polys, all_vars, order='lex')
    
    # 5. Filter for polynomials containing ONLY y
    # Since we used lex with params > y, the polynomials at the end of the basis
    # should depend only on y.
    
    invariants = []
    param_set = set(params)
    
    for p in gb:
        # Check if any parameter is in the free symbols
        # Note: GB elements are poly-like, need .free_symbols (if Expr) or gens check (if Poly)
        # sympy.groebner returns Poly or Expr depending on usage? It returns a GrobnerBasis object which iterates Expr usually.
        # But let's check.
        if isinstance(p, sympy.Poly):
            syms = set(p.gens)
        else:
            syms = p.free_symbols

        if not (syms & param_set):
            invariants.append(p)
            
    return invariants, y_syms
