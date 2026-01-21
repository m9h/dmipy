import pytest
import dmipy_jax.optimization.algebraic_oed as aoed

def test_algebraic_identifiability():
    print("Testing Algebraic Identifiability...")
    
    # Model: 2-Compartment (Bi-Exponential)
    # Unknowns: w1, X1, w2, X2 (4 parameters)
    # We need at least 4 measurements (b-values).
    
    # Case 1: Identifiable (Rich Protocol)
    # 4 shells + b0
    b_identifiable = [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
    result_id = aoed.check_protocol_identifiability(b_identifiable, n_compartments=2)
    print(f"Protocol {b_identifiable}: Identifiable? {result_id}")
    
    # Case 2: Unidentifiable (Too few shells)
    # Only 2 shells + b0 (3 measurements) for 4 unknowns
    b_unidentifiable = [0.0, 1000.0, 2000.0] 
    result_unid = aoed.check_protocol_identifiability(b_unidentifiable, n_compartments=2)
    print(f"Protocol {b_unidentifiable}: Identifiable? {result_unid}")
    
    assert result_id == True, "Expected rich protocol to be identifiable"
    assert result_unid == False, "Expected poor protocol to be unidentifiable"
    
    print("✅ SUCCESS: Algebraic Engine correctly classified protocols.")

if __name__ == "__main__":
    test_algebraic_identifiability()
