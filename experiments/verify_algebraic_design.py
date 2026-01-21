
import sympy
from dmipy_jax.algebra.identifiability import construct_polynomial_system, analyze_identifiability
import time

def verify_algebraic_design():
    print("=== Algebraic Protocol Verification Experiment ===\n")

    # 1. Complexity Minimization
    print("1. Complexity Comparison (Protocol A vs B)")
    
    # Protocol A: Integers [0, 1000] -> X^0, X^1
    proto_A = [0.0, 1000.0]
    
    # Protocol B: Less structured [0, 1300] but scaled to 100 base -> 0, 13
    proto_B = [0.0, 1300.0]
    
    y = [sympy.Symbol(f'y{i}') for i in range(2)]
    
    print(f"Analyzing Protocol A: {proto_A}")
    polys_A, vars_A = construct_polynomial_system(proto_A, y, n_compartments=1)
    res_A = analyze_identifiability(polys_A, vars_A)
    print(f"  Basis Length: {res_A['basis_length']}")
    print(f"  Max Degree: {max(res_A['basis_degrees']) if res_A['basis_degrees'] else 0}")
    
    print(f"Analyzing Protocol B: {proto_B}")
    # Force base calculation logic in construct_polynomial_system to pick small base if integers
    # Ideally implementation picks GCD. 1000, 2000 -> GCD 1000. 1300, 2500 -> GCD 100.
    polys_B, vars_B = construct_polynomial_system(proto_B, y, n_compartments=1)
    res_B = analyze_identifiability(polys_B, vars_B)
    print(f"  Basis Length: {res_B['basis_length']}")
    print(f"  Max Degree: {max(res_B['basis_degrees']) if res_B['basis_degrees'] else 0}")
    
    # Expectation: Protocol A should have lower max degree (e.g. 2 vs 25)
    
    # 2. Degeneracy Counting
    print("\n2. Degeneracy Counting (Bi-Exponential)")
    
    # Underdetermined: 2 shells for 4 vars
    proto_under = [0.0, 1000.0, 2500.0] # 3 measurements total
    y_under = [sympy.Symbol(f'yu{i}') for i in range(3)]
    
    polys_u, vars_u = construct_polynomial_system(proto_under, y_under, n_compartments=2)
    res_u = analyze_identifiability(polys_u, vars_u) # Takes longer
    print(f"  Underdetermined Basis Length: {res_u['basis_length']}")
    print(f"  Is Trivial (1 in GB): {res_u['is_trivial']}")
    # If non-trivial and length > 0, we have an ideal.
    
    # 3. Explicit Inversion (Mono)
    print("\n3. Symbolic Inversion Check")
    # Basis for Mono: {w1 - y0, y0*X1 - y1} (lex order)
    # Check if we can recover w1, X1
    basis_strs = res_A['basis']
    print(f"  Mono Basis Strings: {basis_strs}")
    
    # Simple check if w_1 - y0 is present
    has_S0 = any('w_1 - y0' in str(b) or 'w_1 - y0' in str(b) for b in basis_strs)
    print(f"  Contains 'w_1 - y0' or equivalent? {has_S0}")

if __name__ == "__main__":
    verify_algebraic_design()
