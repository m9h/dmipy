
import sympy
from dmipy_jax.algebra.identifiability import construct_polynomial_system, analyze_identifiability

def main():
    print("Analyzing Bi-Exponential Identifiability with Grobner Bases...")
    
    # 1. Define Theoretical Measurements
    # We assume noiseless signal $meas_k$
    # We want to see if we can recover unique parameters from symbolic measurements.
    
    # Symbols for measurements
    y = [sympy.Symbol(f'y_{k}') for k in range(4)]
    b_values = [0.0, 1.0, 2.0, 3.0] # Sufficient for 4 unknowns?
    
    # 2. Construct System
    # 2 compartments
    polynomials, variables = construct_polynomial_system(b_values, y, n_compartments=2)
    
    print("Variables:", variables)
    print("Polynomials:")
    for p in polynomials:
        print(f"  {p} = 0")
        
    # 3. Compute Grobner Basis
    # Solving for {w1, w2, X1, X2}
    print("\nComputing Grobner Basis (lex order)...")
    res = analyze_identifiability(polynomials, variables)
    
    gb = res['basis']
    print(f"\nGrobner Basis ({len(gb)} polys):")
    for p in gb:
        print(f"  {p}")
        
    # 4. Interpret Results
    # If the system is identifiable up to permutation, we expect polynomials in
    # elementary symmetric functions of X_i?
    # Or quadratic equations for X_i?
    # e.g. a poly like X1^2 - (X1+X2)X1 + X1X2 = 0
    
    # Check if we have equations determining sums/products
    # We can check if X_1 is uniquely determined or if it satisfies a degree-2 polynomial.
    
    # Note: If GB has leading term X_1^2 (not X_1), it means X_1 is not structurally unique (2 solutions).
    
if __name__ == "__main__":
    main()
