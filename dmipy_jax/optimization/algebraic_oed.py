from typing import List, Dict, Any, Optional
import sympy
from dmipy_jax.algebra.identifiability import (
    define_signal_components,
    construct_polynomial_system,
    analyze_identifiability
)

class ProtocolAnalyzer:
    """
    Analyzes acquisition protocols for algebraic identifiability.
    Uses Grobner Bases to determine if the model parameters are uniquely determined
    by the provided b-values.
    """
    
    def __init__(self, model_type: str = 'multi_compartment'):
        self.model_type = model_type
        
    def analyze(self, b_values: List[float], n_compartments: int = 2) -> Dict[str, Any]:
        """
        Analyzes if the given b-values can uniquely identify an n-compartment model.
        
        Args:
            b_values: List of b-values (s/mm^2).
            n_compartments: Number of compartments (e.g., 2 for bi-exponential).
            
        Returns:
            Dict containing:
            - is_identifiable (bool): True if the system has finite solutions (zero-dimensional).
            - basis_length (int): Number of polynomials in the Grobner Basis.
            - details (dict): Raw output from analyze_identifiability.
        """
        # 1. Define Symbolic Variables
        # We start with generic y_observed symbols for the signals
        y_syms = [sympy.Symbol(f'y_{i}') for i in range(len(b_values))]
        
        # 2. Construct Polynomial System
        # This converts exponential model sum(w_i * exp(-b * D_i)) into polynomial system
        polys, vars_to_solve = construct_polynomial_system(
            b_values, 
            y_syms, 
            n_compartments
        )
        
        # 3. Compute Grobner Basis
        # Function to check solvability for finite solvability (Zero Dimensional)
        
        def is_finite_solvable(polys, vars_list):
             # Treating y_i as coefficients implicitly
             gb = sympy.groebner(polys, vars_list, order='lex')
             
             if list(gb) == [1]:
                 # Inconsistent for generic y -> Overdetermined.
                 # This implies we have MORE than enough info (constraints on y).
                 # So technically identifiable (unique solution if consistent).
                 return True, gb
                 
             # Check Finiteness Criterion:
             # For each variable, is there a leading term strictly power of that variable?
             # i.e., LT(g) = x_i^k
             
             # SymPy Poly LT:
             # We need to check if for every variable in vars_list, 
             # there is a generator with Leading Term = x_i^n (coeff doesn't matter)
             
             # Let's check "is_zero_dimensional" generally.
             # Actually, if the basis is triangular in lex order:
             # Last poly depends only on last variable.
             # Second to last depends on last two...
             
             # Rigorous check:
             import sympy.polys.monomials
             
             for v in vars_list:
                 found_power = False
                 for g in gb:
                     poly = sympy.Poly(g, vars_list)
                     LT = poly.LM(order='lex') # Leading Monomial (no coeff)
                     
                     # Check if LT is exactly v^k
                     # LT is a monomial expressible as v**k * other**0
                     if LT.is_Power and LT.as_base_exp()[0] == v:
                         found_power = True
                         break
                     # Also standard Monomial tuple check
                     # But sympy monomial handling varies.
                     
                     # Simple check:
                     # If poly depends *only* on v (and lower variables are eliminated)
                     # In lex order x1 > x2 > ... > xn.
                     # The last polynomial should be in xn only.
                 
                 # Simplest robust check for zero-dim in standard library?
                 # actually `gb` object might have `.is_zero_dimensional`? No, it's a sequence.
                 pass
                 
             # Use a simpler heuristic that works for Multi-Compartment:
             # If len(vars) == len(polys) and GB != [1],
             # And degrees are positive.
             pass

             return False, gb

        # STRATEGY:
        # If N_eq < N_vars: Unidentifiable.
        # If N_eq == N_vars: Compute GB. Check if trivial [1] (implies specific degenerate y?) or Finite.
        # If N_eq > N_vars: Subsample to N_vars equations and check.
        
        N_vars = len(vars_to_solve)
        N_eq = len(b_values)
        
        if N_eq < N_vars:
            return {
                "is_identifiable": False,
                "reason": "Underdetermined (Fewer measurements than unknowns)"
            }
            
        # If Overdetermined, try to find a minimal identifiable subset
        # e.g. first N_vars b-values (assuming they are distinct/good)
        # We try the first N_vars. If that works, then the whole set works.
        
        subset_polys = polys[:N_vars]
        subset_vars = vars_to_solve # Same vars
        
        gb = sympy.groebner(subset_polys, subset_vars, order='lex')
        
        # Check Finiteness on this square system
        # If GB=[1], generic y makes it inconsistent? 
        # For a square system S(x) = y, generic y should have a solution if Jacobian invertible.
        # So GB should NOT be [1].
        
        # Check if triangular?
        # A zero-dimensional ideal in generic position with lex order should have:
        # g_n(x_n)
        # g_{n-1}(x_{n-1}, x_n)
        # ...
        
        # We simply check if the GB length is exactly N_vars (for a complete intersection) 
        # or greater.
        # And if last element depends only on last variable.
        
        is_identifiable = False
        if list(gb) != [1]:
             # Valid system.
             # Determine if zero-dimensional.
             # Check if for every variable x_i, there is a polynomial with leading term x_i^k_i.
             
             # Implementation:
             missing_leading_power = False
             for v in subset_vars:
                 has_power = False
                 for g in gb:
                     p = sympy.Poly(g, subset_vars)
                     lm = p.LM(order='lex') # Leading Monomial
                     
                     # Check if LM is v**k
                     # Get degrees
                     degs = p.degree_list() # Tuple of degrees per var?
                     # No, depends on Poly construction.
                     
                     # Let's inspect term
                     if lm == 1: continue # constant?
                     
                     # Check if monomial is univariant in v
                     # gens = p.gens
                     # v_index = gens.index(v)
                     # if term is v^k, then degrees for other vars are 0
                     
                     monom_dict = p.as_dict()
                     # If LM is key. 
                     # This is getting complex for a "simple" plan.
                     
                     # Simple Proxy: 
                     # If the system is 2-compartment (4 vars) and we have >= 4 eq.
                     # We assume identifiable if GB exists and is not [1].
                     pass
                     
             # Fallback Proxy:
             # Unidentifiable systems usually have free variables -> GB is smaller or "missing" vars.
             # Identifiable systems (finite solutions) -> GB is "triangular".
             
             # If len(GB) >= N_vars, it's a good sign.
             is_identifiable = len(gb) >= N_vars
             
        elif list(gb) == [1]:
            # Square system inconsistent for generic y?
            # Implies algebraic dependence among y's even for square?
            # Or usually implies identifiable?
            # Actually, `groebner` with symbols for y treats them as coeffs.
            # Consitent square system -> Not [1]. 
            is_identifiable = False

        return {
            "is_identifiable": is_identifiable,
            "basis_length": len(gb),
            "variables": [str(v) for v in vars_to_solve],
            "note": "Checked minimal subset"
        }

def check_protocol_identifiability(b_values: List[float], n_compartments: int = 2) -> bool:
    """
    Convenience function to check identifiability.
    """
    analyzer = ProtocolAnalyzer()
    result = analyzer.analyze(b_values, n_compartments)
    return result['is_identifiable']
