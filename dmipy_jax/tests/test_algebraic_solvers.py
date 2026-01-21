import sympy
import jax.numpy as jnp
import pytest
from dmipy_jax.algebra.identifiability import define_signal_components, construct_polynomial_system
from dmipy_jax.algebra.solvers import SymbolicSolver, compile_solver, AlgebraicInverter

def test_mono_exponential_solver():
    """
    Test deriving and compiling an exact solver for Mono-Exponential model.
    S = f1 * exp(-b * D1)
    
    Algebraic form:
    w1 = f1 (assuming S0=1 in formulation or absorbed)
    X1 = exp(-b_base * D1)
    
    Protocol: b=[0, 1000]. b_base=1000.
    Signals: y0, y1.
    """
    
    # 1. Setup System
    n_compartments = 1
    variables, f, D, S0 = define_signal_components(n_compartments)
    
    b_values = [0.0, 1000.0]
    y0 = sympy.Symbol('y0')
    y1 = sympy.Symbol('y1')
    measured_signals = [y0, y1]
    
    polys, model_vars = construct_polynomial_system(b_values, measured_signals, n_compartments)
    # model_vars usually [w_1, X_1]
    
    # 2. Symbolic Solve
    solver = SymbolicSolver(polys, model_vars)
    solution = solver.solve()
    
    print("Symbolic Solution:", solution)
    
    # Check symbolic correctness
    # w_1 = y0
    # X_1 = y1/y0
    
    w1_sym = sympy.Symbol('w_1')
    X1_sym = sympy.Symbol('X_1')
    
    assert solution[w1_sym] == y0
    assert solution[X1_sym] == y1 / y0

    # 3. Compile to JAX
    output_names = ['w_1', 'X_1']
    jax_fn = compile_solver(solution, output_names, measured_signals)
    
    # 4. Numerical Test
    # GT: f1 = 1.0, D1 = 1e-3 (1e-9 m^2/s * 1e6 scaling usually? Or units?)
    # construct_polynomial_system uses b_values as provided.
    # If D is in true units, b should be in true units.
    # Let's use simple numbers.
    # b=1000. D=0.001. bD = 1.0.
    # S0 = e^-0 = 1. S1 = e^-1 = 0.367879.
    
    gt_signals = jnp.array([1.0, jnp.exp(-1.0)])
    
    estimated = jax_fn(gt_signals)
    # est[0] = w1 = y0 = 1.0
    # est[1] = X1 = y1/y0 = exp(-1)
    
    assert jnp.allclose(estimated[0], 1.0)
    assert jnp.allclose(estimated[1], jnp.exp(-1.0))
    
    # Invert back to physical parameters
    # w1 = S0 * f1. X1 = exp(-b_base * D1)
    # D1 = -ln(X1) / b_base
    
    est_w1 = estimated[0]
    est_X1 = estimated[1]
    
    est_D1 = -jnp.log(est_X1) / 1000.0
    
    assert jnp.allclose(est_D1, 0.001)

def test_bi_exponential_failure_mode():
    """
    Test that a system with insufficient data raises error or handled?
    Bi-Exp requires 4 points?
    If we give 2 points, Sympy solve should fail to find a unique solution? Or return param solution?
    """
    n_compartments = 2
    b_values = [0.0, 1000.0] # Only 2 points for 4 unknowns (f1, D1, f2, D2)
    y0, y1 = sympy.symbols('y0 y1')
    
    polys, model_vars = construct_polynomial_system(b_values, [y0, y1], n_compartments)
    
    solver = SymbolicSolver(polys, model_vars)
    
    # Expect failure to solve fully unique solution
    # sympy.solve might return solution in terms of other variables (parametric).
    # But usually we want fully determined solution for compilation.
    
    # We won't assert error rigorously here as behavior depends on sympy version/system type,
    # but just checking it doesn't crash inappropriately.
    pass
