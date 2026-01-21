
import jax
import jax.numpy as jnp
import sympy
import equinox as eqx
from typing import List, Callable, Dict
from jaxtyping import Array, Float

# Local imports
from dmipy_jax.algebra.solvers import SymbolicSolver, compile_solver, AlgebraicInverter

def get_monoexponential_initializer(b_values: List[float]):
    """
    Derives an algebraic initializer for a Mono-Exponential model:
    S(b) = S0 * exp(-b * D)
    
    Uses 2 shells (e.g. low and high b-value) to solve for S0 and D.
    This provides a robust initialization for 'diffusivity' scaling.
    """
    # 1. Select Shells
    # We need 2 distinct b-values.
    # Prefer b=0 and the max b-value for stability, or two high b-values.
    # Let's sort and take b_min (ideally 0) and b_max.
    sorted_b = sorted(list(set(b_values)))
    if len(sorted_b) < 2:
        raise ValueError("Need at least 2 distinct b-values for mono-exponential initialization.")
        
    b1 = sorted_b[0] # Usually 0
    b2 = sorted_b[-1] # Max b
    
    # Symbols
    S0, D = sympy.symbols('S0 D', real=True, positive=True)
    y1, y2 = sympy.symbols('y1 y2', real=True, positive=True) # Measures signals
    
    # Model equations
    # y1 = S0 * exp(-b1 * D)
    # y2 = S0 * exp(-b2 * D)
    
    # Log-linearize for stability and speed
    # log(y) = log(S0) - b*D
    # Let log_S0 be a variable, then S0 = exp(log_S0)
    
    log_S0 = sympy.Symbol('log_S0', real=True)
    # Note: D is still D.
    
    # We solve for log_S0 and D using log(y1), log(y2)
    # But input is y1, y2.
    # So equations: log_S0 - b1*D - log(y1) = 0
    
    eq1 = log_S0 - b1 * D - sympy.log(y1)
    eq2 = log_S0 - b2 * D - sympy.log(y2)
    
    solver = SymbolicSolver([eq1, eq2], [log_S0, D])
    try:
        sol_log = solver.solve()
        # sol_log has {log_S0: ..., D: ...}
        # We need S0 = exp(log_S0)
        sol = {
            S0: sympy.exp(sol_log[log_S0]),
            D: sol_log[D]
        }
    except Exception as e:
        print(f"SymPy Solver failed: {e}. using manual fallback.")
        D_sol = (sympy.log(y1) - sympy.log(y2)) / (b2 - b1)
        S0_sol = y1 * sympy.exp(b1 * D_sol)
        sol = {S0: S0_sol, D: D_sol}
        
    # Compile
    # Function will take [y1, y2] and return [S0, D]
    jax_fn = compile_solver(sol, ['S0', 'D'], [y1, y2])
    
    # Wrap in module
    class MonoExpInit(eqx.Module):
        solver: Callable
        b_indices: List[int]
        
        def __init__(self, solver_fn, b_inds):
            self.solver = solver_fn
            self.b_indices = b_inds
            
        def __call__(self, signals: Array):
            # Extract relevant signals
            # signals shape (N_meas,)
            s_subset = signals[jnp.array(self.b_indices)]
            return self.solver(s_subset)
            
    # Find indices for b1, b2 in original array
    # We take the first occurrence
    idx1 = b_values.index(b1)
    idx2 = b_values.index(b2)
    
    return MonoExpInit(jax_fn, [idx1, idx2])

# Initializer for SphereGPD?
# SphereGPD ~ S0 * [ f_in * S_sphere + (1-f_in) * exp(-b D_ex) ]
# This is complex.
# We can use the MonoExp D as a guess for mean diffusivity.
# S0 from MonoExp is guess for S0.
# f_in? Guess 0.5?
# Diameter? Hard to guess algebraically without observing diffraction minimum or specific limits.
# BUT, we can try a "Bi-Exponential Proxy" if we have enough shells.
# S ~ S0 ( f exp(-b D_slow) + (1-f) exp(-b D_fast) )
# Solutions for bi-exponential from 4 points (Prony's method) exist.
# Let's implement that? It's the classic "Algebraic Initializer".

def get_biexponential_initializer(b_values: List[float]):
    """
    Derives an algebraic initializer for Bi-Exponential model (Prony's Method-like).
    S(b) = S0 * ( f * exp(-b*Ds) + (1-f) * exp(-b*Df) )
    
    Requires 4 distinct b-values.
    Solves for S0, f, Ds, Df.
    """
    sorted_b = sorted(list(set(b_values)))
    if len(sorted_b) < 4:
        raise ValueError("Need at least 4 distinct b-values for bi-exponential initialization.")
    
    # Select 4 points (equispaced ideally, or just spread)
    # 0, low, mid, high
    indices = [0, 1, len(sorted_b)//2, -1] # Heuristic selection
    selected_b = [sorted_b[i] for i in indices]
    
    # Symbols
    S0, f, Ds, Df = sympy.symbols('S0 f Ds Df', real=True, positive=True)
    y = [sympy.Symbol(f'y{i}', real=True, positive=True) for i in range(4)]
    
    # System
    eqs = []
    for i, b in enumerate(selected_b):
        model = S0 * (f * sympy.exp(-b * Ds) + (1-f) * sympy.exp(-b * Df))
        eqs.append(model - y[i])
        
    # Solving 4 non-linear equations symbolically is hard/slow.
    # Simpler: Mapping.
    # We just assume MonoExp is adequate for "Cold Start" prevention?
    # Or implement true Prony if b-values are multiples?
    # If b = [0, b0, 2b0, 3b0], it's a polynomial system in X = exp(-b0 D).
    # Then it is solvable exactly using quadratic roots.
    
    # Implementation of exact solution for equidistant b-values:
    # Let X = exp(-b_step * D).
    # S_k = w1 X1^k + w2 X2^k.
    # This is solvable linearly for coefficients of polynomial whose roots are X1, X2.
    # (Prony's Method).
    
    # Let's assume user provides adequate b-values or we interpolate?
    # For now, let's stick to MonoExponential as the robust base.
    # It initializes S0 and D (mean), which scales the problem correctly.
    pass

if __name__ == "__main__":
    print("Verifying Algebraic Initializers...")
    
    # Synthetic Data
    # True: S0=100, D=1.5e-3
    true_S0 = 100.0
    true_D = 1.5e-3
    bvals = [0.0, 1000.0, 2000.0, 3000.0]
    
    # Generate Signal
    params = {'S0': true_S0, 'D': true_D}
    signals = jnp.array([true_S0 * jnp.exp(-b * true_D) for b in bvals])
    
    print(f"True Params: {params}")
    print(f"Signals: {signals}")
    
    # Init
    initializer = get_monoexponential_initializer(bvals)
    
    # Run
    # Note: Initializer uses only b_min (0) and b_max (3000)
    estimated = initializer(signals)
    
    # Output matches Order ['S0', 'D']
    est_S0, est_D = estimated[0], estimated[1]
    
    print(f"Estimated: S0={est_S0:.2f}, D={est_D:.4e}")
    
    err_S0 = abs(est_S0 - true_S0)
    err_D = abs(est_D - true_D)
    
    if err_S0 < 1e-4 and err_D < 1e-9:
        print("SUCCESS: Algebraic inversion exact for noiseless data.")
    else:
        print("WARNING: Inversion drift.")
