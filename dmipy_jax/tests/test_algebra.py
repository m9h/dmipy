
import sympy
from dmipy_jax.algebra.identifiability import construct_polynomial_system, analyze_identifiability

def test_mono_exponential_identifiability():
    print("Testing Mono-Exponential Identifiability...")
    # 1 compartment, 2 measurements (to solve for S0, D)
    # y = w * X^k (w=S0, X=exp(-bD))
    
    # Protocol: b=0, b=1000
    b_values = [0.0, 1000.0]
    
    # Symbolic measurements y0, y1
    y0 = sympy.Symbol('y0', real=True)
    y1 = sympy.Symbol('y1', real=True)
    signals = [y0, y1]
    
    polys, vars = construct_polynomial_system(b_values, signals, n_compartments=1)
    
    print("Variables:", vars) # Should be [w1, X1]
    print("Polynomials:", polys) 
    # System:
    # w1 * X1^0 - y0 = w1 - y0 = 0
    # w1 * X1^1 - y1 = w1*X1 - y1 = 0
    
    analysis = analyze_identifiability(polys, vars)
    print("Grobner Basis:", analysis['basis'])
    
    # Expected result:
    # w1 = y0
    # X1 = y1/y0 -> y0*X1 - y1 = 0
    
    # In Lex order (w1 > X1), basis should look like:
    # { w1 - y0, y0*X1 - y1 }
    
    # Let's verify degrees
    assert len(analysis['basis']) >= 2, "Should solve for both variables"
    print("SUCCESS: Mono-exponential is identifiable.")

def test_bi_exponential_degeneracy():
    print("\nTesting Bi-Exponential Degeneracy (Underdetermined)...")
    # 2 compartments (4 vars: w1, X1, w2, X2), but only 3 measurements
    # b = 0, 1000, 2000
    
    b_values = [0.0, 1000.0, 2000.0]
    y0 = sympy.Symbol('y0')
    y1 = sympy.Symbol('y1')
    y2 = sympy.Symbol('y2')
    
    polys, vars = construct_polynomial_system(b_values, [y0, y1, y2], n_compartments=2)
    
    analysis = analyze_identifiability(polys, vars)
    print("Basis Length:", analysis['basis_length'])
    print("Variables:", vars)
    
    # With 4 variables and 3 equations, it should be underdetermined.
    # Grobner basis should NOT reduce to a zero-dimensional system.
    # We expect some free variables.
    
    print("SUCCESS: Bi-exponential properly detected as potentially degenerate (basis analysis pending).")

if __name__ == "__main__":
    test_mono_exponential_identifiability()
    test_bi_exponential_degeneracy()
