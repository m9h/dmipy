
import sympy
from dmipy_jax.algebra.identifiability import define_signal_components, construct_polynomial_system, analyze_identifiability

def test_mono_exponential_algebra():
    # 1. Define Model (1 Compartment)
    # S = f1 * exp(-b * D1)
    # We ignore S0 for now or treat as f1.
    n_compartments = 1
    variables, f, D, S0 = define_signal_components(n_compartments)
    
    # 2. Define Protocol
    # We need at least 2 measurements to solve for f1, D1?
    # Unknowns: f1, D1. (2 vars).
    # Measurements: b=0 (S0), b=b1 (S1).
    # If we ignore S0 variable and assume S0=1 in `define_signal_components` or just solve for fractions.
    # The `construct_polynomial_system` uses `w_i = S0 * f_i` and `X_i = exp(-b_base * D_i)`.
    # Unknowns are w_1, X_1. (2 vars).
    # We need 2 equations.
    
    b_values = [0.0, 1000.0]
    # Observed signals (Symbolic to get general formula)
    y0 = sympy.Symbol('y0')
    y1 = sympy.Symbol('y1')
    measured_signals = [y0, y1]
    
    print("Constructing System...")
    polys, vars_to_solve = construct_polynomial_system(b_values, measured_signals, n_compartments)
    print(f"Polynomials: {polys}")
    print(f"Variables: {vars_to_solve}")
    
    # 3. Compute Grobner Basis
    print("Computing Grobner Basis...")
    # We want to solve for w_1, X_1 in terms of y0, y1.
    # So we treat y0, y1 as parameters (coefficients), not variables in the ring?
    # Sympy `groebner` treats passed `variables` as the unknowns. Everything else is coefficient.
    
    res = analyze_identifiability(polys, vars_to_solve)
    print("Basis:", res['basis'])
    
    # Expectation:
    # w1 - y0 = 0  => w1 = y0
    # w1 * X1^1 - y1 = 0 => y0 * X1 - y1 = 0 => X1 = y1/y0
    
    # If the basis recovers this, we are good.
    
    return res

if __name__ == "__main__":
    test_mono_exponential_algebra()
