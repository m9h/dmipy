import sympy
import jax
import jax.numpy as jnp
from typing import List, Dict, Callable, Union, Sequence

class SymbolicSolver:
    """
    Solves a polynomial system (defined by a Grobner Basis or list of polynomials) 
    symbolically for the target variables.
    """
    def __init__(self, polynomials: List[sympy.Expr], variables: List[sympy.Symbol]):
        """
        Args:
            polynomials: List of sympy expressions (equation = 0).
            variables: List of variables to solve for (unknowns).
        """
        self.polynomials = polynomials
        self.variables = variables

    def solve(self) -> Dict[sympy.Symbol, sympy.Expr]:
        """
        Attempts to solve the system for the variables.
        
        Returns:
            Dictionary mapping target variables to solutions (in terms of parameters/signals).
            Example: {w_1: y0, X_1: y1/y0}
        """
        # sympy.solve returns a list of dictionaries if dict=True
        solutions = sympy.solve(self.polynomials, self.variables, dict=True)
        
        if not solutions:
            raise ValueError("No symbolic solution found for the polynomial system.")
            
        # For now, we take the first solution. 
        # In multi-root cases (e.g. crossing fibers), we might need to handle multiplicity.
        # But usually 'identifiable' theoretical models implies unique solution in feasible region?
        return solutions[0]

def compile_solver(
    solution: Dict[sympy.Symbol, sympy.Expr], 
    output_names: List[str], 
    input_syms: List[sympy.Symbol]
) -> Callable:
    """
    Compiles a symbolic solution into a JAX-compatible function.
    
    Args:
        solution: Dictionary of solutions {Symbol('w1'): Expr, ...}.
        output_names: Ordered list of string names for output parameters (e.g. ['w_1', 'X_1']).
        input_syms: Ordered list of sympy symbols representing input arguments (e.g. [y0, y1]).
        
    Returns:
        A JAX function: func(inputs) -> jnp.array([out1, out2, ...])
    """
    
    # 1. Prepare Ordered Output Expressions
    output_exprs = []
    
    # We need to match output_names to keys in solution dictionary
    # Convert solution keys to strings for mapping
    sol_map = {str(k): v for k, v in solution.items()}
    
    for name in output_names:
        if name not in sol_map:
            raise ValueError(f"Output parameter '{name}' not found in solution keys: {list(sol_map.keys())}")
        output_exprs.append(sol_map[name])
        
    # 2. Lambdify using JAX backend
    # We instruct SymPy to use 'jax' (if available) or 'numpy' which JAX can trace (often safer).
    # Since we want it to work inside JIT, using 'jax' module strings is best.
    
    # Note: sympy.lambdify(args, exprs, 'jax')
    jax_func_raw = sympy.lambdify(input_syms, output_exprs, modules='jax')
    
    # 3. Create Wrapper
    def solver_fn(inputs: Union[jnp.ndarray, Sequence[float]]):
        """
        Args:
            inputs: Array or sequence of signal values corresponding to input_syms.
        Returns:
            Array of estimated parameters.
        """
        # Unpack inputs because lambdify expects *args
        # Convert to list/tuple if array
        if hasattr(inputs, 'tolist'):
           args = inputs.tolist()
        else:
           args = inputs
           
        # Call generated function
        # It returns a list of results (if output_exprs is list) or tuple
        results = jax_func_raw(*args)
        
        return jnp.array(results)
        
    return solver_fn

class AlgebraicInverter:
    """
    High-level class to hold a compiled algebraic inverse model.
    """
    def __init__(self, solver_fn: Callable, output_names: List[str]):
        self._solver = solver_fn
        self.output_names = output_names
        
    def __call__(self, signals):
        return self._solver(signals)
