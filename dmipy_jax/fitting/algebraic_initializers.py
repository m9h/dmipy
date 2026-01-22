
import jax
import jax.numpy as jnp
import equinox as eqx
import sympy
from typing import List, Callable, Dict, Any
from dmipy_jax.algebra.identifiability import construct_polynomial_system

def sympy_to_jax_func(expr: sympy.Expr, input_vars: List[sympy.Symbol]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Converts a SymPy expression into a JIT-able JAX function.
    
    Args:
        expr: The symbolic expression (e.g., y1/y0).
        input_vars: List of symbols representing the input array (e.g. [y0, y1]).
        
    Returns:
        A function f(inputs) -> result
    """
    # Use sympy.lambdify with "jax" backend
    # Note: 'jax' backend in lambdify maps to jax.numpy
    
    # We create a wrapper that unpacks the input array
    # because lambdify expects arguments matching input_vars
    
    lam = sympy.lambdify(input_vars, expr, modules=["jax"])
    
    def wrapped(inputs):
        # inputs: (N_measurements,)
        # unpack
        # Note: if N is large, this unpacking might be slow in python trace, 
        # but JAX trace handles it.
        # Ideally we pass individual args if possible, but standard interface is array.
        return lam(*inputs)
        
    return wrapped

class AlgebraicInitializer(eqx.Module):
    """
    A trainable (or fixed) initializer derived from algebraic inversion.
    """
    funcs: List[Callable] = eqx.field(static=True)
    param_names: List[str] = eqx.field(static=True)
    
    def __init__(self, funcs: List[Callable], param_names: List[str]):
        self.funcs = tuple(funcs)
        self.param_names = tuple(param_names)
        
    def __call__(self, signals: jnp.ndarray) -> Dict[str, float]:
        """
        Args:
           signals: Array of signal measurements corresponding to the protocol used for derivation.
        Returns:
           Dict of parameter estimates.
        """
        results = {}
        for name, fn in zip(self.param_names, self.funcs):
            results[name] = fn(signals)
        return results

def derive_rational_solution(
    b_values: List[float], 
    n_compartments: int
) -> AlgebraicInitializer:
    """
    Derives the algebraic solution for a protocol and returns an AlgebraicInitializer.
    
    Currently supports Mono-Exponential (N=1) on 2 shells.
    Future: Use Elimination Ideal to solve N=2.
    """
    measurements = [sympy.Symbol(f'y{i}', real=True) for i in range(len(b_values))]
    
    # 1. Construct Polynomial System
    polys, vars_all = construct_polynomial_system(b_values, measurements, n_compartments)
    # vars_all = [w1, X1, (w2, X2...)]
    
    # 2. Compute Grobner Basis
    # We want to solve for w_i and X_i in terms of y.
    # We use Lex order with y as "constants" (lowest in order)?
    # Actually, in construct_polynomial_system, y are in the coefficients (from SymPy's perspective of generators?)
    # Wait, 'y' are Symbols. If we include them in generators, we can't solve "in terms of y".
    # We should treat 'y' as coefficients (params).
    # Since construct_polynomial_system puts them in the expression, `sympy.groebner` needs to know 
    # the variables we want to eliminate/solve for.
    # `vars_all` contains the model parameters. `measurements` are not in `vars_all`.
    # So `sympy.groebner(polys, vars_all)` treats `measurements` as symbolic constants. Perfect.
    
    gb = sympy.groebner(polys, vars_all, order='lex')
    
    # 3. Extract Solutions
    # For Mono-Exp (N=1, 2 shells):
    # Basis: [w1 - y0, X1*y0 - y1]
    # Sol: w1 = y0, X1 = y1/y0
    
    # We assume the basis is triangular and linear in the variables we want (Rational Identifiability).
    # If degree > 1, we might need root finding (not purely rational).
    
    solutions = {}
    
    # Reverse iteration to back-substitute (if needed)
    # For now, simple parsing of linear terms: a*X - b = 0 -> X = b/a
    
    params_found = []
    funcs = []
    
    for v in vars_all:
        # Find polynomial where v is the leading term
        found_poly = None
        for p in gb:
            if sympy.degree(p, v) == 1 and all(sympy.degree(p, other) == 0 for other in vars_all if other != v and vars_all.index(other) < vars_all.index(v)):
                 # This is a bit strict for lex, usually just check leading term is exactly v^1 * possible_coeffs
                 # Let's simplfy: find p where v is present, and only variables appearing are v and 'later' variables (which are known/solved if iterating backwards).
                 # Wait, Lex order: w1 > X1.
                 # equation for X1 involves just X1 (and y).
                 # equation for w1 involves w1 and X1 (and y).
                 found_poly = p
                 break
        
        # Actually, let's just solve the system using sympy.solve which is easier for extraction
        # once (approx) triangular.
        # Or better: `sympy.solve(gb, vars_all)`
        
        sols = sympy.solve(gb, vars_all, dict=True)
        if not sols:
            raise ValueError("No solution found or system is inconsistent/degenerate.")
            
        # Assuming unique generic solution
        sol = sols[0] # Take first branch
        
        # Setup parameter names
        # w_i = S0 * f_i
        # X_i = exp(-b_base * D_i) -> D_i = -ln(X_i)/b_base
        
        # We want to return [w1, D1, ...]
        # Note: derive_rational_solution returns JAX funcs for physical parameters (f, D).
        
    # Re-map w, X to physical f, D
    # We need b_base used in construction
    non_zero_b = [b for b in b_values if b > 1e-6]
    b_base = min(non_zero_b) if non_zero_b else 1.0 # fallback
    
    # w_i = f_i (assuming S0 handled or sum(w)=S0)
    # Let's assume w_i are the effective signal fractions.
    
    final_funcs = []
    final_names = []
    
    # Extract w_i and X_i from solution
    for i in range(1, n_compartments + 1):
        w_sym = sympy.Symbol(f'w_{i}')
        X_sym = sympy.Symbol(f'X_{i}')
        
        if w_sym in sol and X_sym in sol:
            # Fraction w
            expr_w = sol[w_sym]
            final_funcs.append(sympy_to_jax_func(expr_w, measurements))
            final_names.append(f'f_{i}')
            
            # Diffusivity D
            # D = -ln(X) / b_base
            expr_X = sol[X_sym]
            expr_D = -sympy.log(expr_X) / b_base
            final_funcs.append(sympy_to_jax_func(expr_D, measurements))
            final_names.append(f'D_{i}')
            
    return AlgebraicInitializer(final_funcs, final_names)

