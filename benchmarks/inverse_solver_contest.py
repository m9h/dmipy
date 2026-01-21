
import time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp
from scico import functional, linop, loss, optimize

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
# JAX 64-bit might be relevant for precision comparison, but usually float32 is fine for MRI
# jax.config.update("jax_enable_x64", True) 

def benchmark_inverse_solver_contest():
    print("=======================================================")
    print("   AMICO SOLVER CONTEST: CVXPY (Legacy) vs SCICO (JAX) ")
    print("=======================================================")

    # ----------------------------------------------------------------
    # 1. Setup Data and Dictionary
    # ----------------------------------------------------------------
    N_voxels = 2000 # Enough to show scaling, not too long for interactive run
    N_atoms = 100   # Size of dictionary
    N_meas = 60     # Number of measurements (gradient directions)
    
    print(f"Configuration: {N_voxels} voxels, {N_atoms} atoms, {N_meas} measurements")

    # Generate random Dictionary Matrix M [N_meas, N_atoms]
    np.random.seed(42)
    M = np.abs(np.random.randn(N_meas, N_atoms))
    # Normalize columns usually? Let's just keep it random.
    # Actually, usually atoms are normalized signals.
    M = M / np.linalg.norm(M, axis=0, keepdims=True)

    # Generate Ground Truth Weights X_gt [N_atoms, N_voxels]
    # Make it sparse
    sparsity = 0.1 # 10% active atoms
    X_gt = np.zeros((N_atoms, N_voxels))
    mask = np.random.rand(N_atoms, N_voxels) < sparsity
    X_gt[mask] = np.random.rand(np.sum(mask)) # Positive weights
    
    # Generate Data Y = M X + Noise
    Y = M @ X_gt # [N_meas, N_voxels]
    # Add some noise
    noise_level = 0.01
    Y += noise_level * np.random.randn(*Y.shape)
    
    # Ensure non-negative data? MRI data is magnitude.
    Y = np.abs(Y)
    
    print("Data generated.")

    # Problem constants
    lambda_l1 = 0.001
    lambda_l2 = 0.0
    
    # ----------------------------------------------------------------
    # 2. CVXPY Implementation (Reference)
    # ----------------------------------------------------------------
    print("\n--- Running CVXPY (Legacy) ---")
    
    # Define single voxel solver function
    def solve_cvxpy_voxel(y_signal, M_matrix):
        x = cp.Variable(N_atoms)
        # Loss: 0.5 * ||Ax - y||^2
        data_fidelity = 0.5 * cp.sum_squares(M_matrix @ x - y_signal)
        # Reg: lambda * ||x||_1
        reg = lambda_l1 * cp.norm(x, 1)
        # Constraints: x >= 0
        prob = cp.Problem(cp.Minimize(data_fidelity + reg), [x >= 0])
        prob.solve() # Let cvxpy choose available solver (likely CLARABEL or SCS)
        return x.value

    t0 = time.time()
    X_cvxpy = []
    # Loop over voxels
    for i in range(N_voxels):
        res = solve_cvxpy_voxel(Y[:, i], M)
        if res is None:
             res = np.zeros(N_atoms)
        X_cvxpy.append(res)
        if (i+1) % 500 == 0:
            print(f"  Processed {i+1} voxels...")
            
    cvxpy_time = time.time() - t0
    X_cvxpy = np.array(X_cvxpy).T # [N_atoms, N_voxels]
    
    print(f"CVXPY Time: {cvxpy_time:.4f} s")
    print(f"CVXPY Throughput: {N_voxels / cvxpy_time:.2f} voxels/s")


    # ----------------------------------------------------------------
    # 3. SCICO Implementation (JAX)
    # ----------------------------------------------------------------
    print("\n--- Running SCICO (JAX) ---")
    
    # Define functional form matching CVXPY
    # f(x) = 0.5 * ||Ax - y||^2
    # g(x) = lambda * ||x||_1 + I(x>=0)
    
    # We will use the dmipy_jax ADMM pattern
    # But implement it directly here to ensure exact comparison control
    # (The class wraps this, but we want to be sure of the setup)
    
    # Convert to device (though JAX does this auto)
    M_jax = jax.device_put(jnp.array(M))
    
    # Define NonNegativeL1 functional for PGM
    class NonNegativeL1(functional.Functional):
        has_eval = True
        has_prox = True
        
        def __init__(self, alpha=1.0):
            self.alpha = alpha
            
        def __call__(self, x):
            # If any x < 0, inf
            # Else alpha * sum(|x|)
            # JAX safe
            is_neg = jnp.any(x < 0)
            return jax.lax.cond(
                is_neg,
                lambda _: jnp.inf,
                lambda _: self.alpha * jnp.sum(jnp.abs(x)),
                operand=None
            )
            
        def prox(self, v, s):
            # Prox of I(x>=0) + alpha * |x|_1
            # = max(0, SoftThreshold(v, s * alpha))
            # SoftThreshold(v, t) = sign(v) * max(|v| - t, 0)
            threshold = s * self.alpha
            st = jnp.sign(v) * jnp.maximum(jnp.abs(v) - threshold, 0)
            return jnp.maximum(0, st)

    # Pre-compute operator
    A_op = linop.MatrixOperator(M_jax)
    
    @jax.jit
    def fit_batch(Y_batch):
        # Y_batch: [N_meas, BatchSize]
        
        # Loss f(x) = 0.5 ||Ax - y||^2
        f = loss.SquaredL2Loss(y=Y_batch, A=A_op)
        
        # Regularizer g(x) = lambda * ||x||_1 + I(x>=0)
        g = NonNegativeL1(alpha=lambda_l1)
        
        # PGM (FISTA) Solver
        # Uses gradient of f and prox of g
        solver = optimize.PGM(
            f=f,
            g=g,
            L0=1.05 * jnp.linalg.norm(M_jax, 2)**2, # Lipschitz constant of gradient (slightly > L)
            x0=jnp.zeros((N_atoms, Y_batch.shape[1])), # Start from 0
            maxiter=200, # Compromise for speed
            itstat_options={'display': False}
        )
        
        return solver.solve()
    
    # Run Benchmark
    # ... code continues ...


    # Prepare Data
    Y_jax = jnp.array(Y) # [N_meas, N_voxels]
    
    # Warmup
    print("  Warming up JAX...")
    _ = fit_batch(Y_jax[:, :10]).block_until_ready()
    print("  Warmup done.")
    
    # Run Benchmark
    t0 = time.time()
    X_scico = fit_batch(Y_jax).block_until_ready()
    jax_time = time.time() - t0
    
    print(f"JAX Time: {jax_time:.4f} s")
    print(f"JAX Throughput: {N_voxels / jax_time:.2f} voxels/s")


    # ----------------------------------------------------------------
    # 4. Results
    # ----------------------------------------------------------------
    
    speedup = cvxpy_time / jax_time
    print(f"\n--- Results ---")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 10:
        print("SUCCESS: >10x Speedup achieved!")
    else:
        print("WARNING: Speedup is less than 10x.")

    # Accuracy Check
    # Compare X_cvxpy and X_scico
    # SCICO returns X [N_atoms, N_voxels]
    
    diff = np.abs(X_cvxpy - np.array(X_scico))
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    # Norm difference relative to signal
    # frobenius norm
    norm_diff = np.linalg.norm(diff) / np.linalg.norm(X_cvxpy)
    
    print(f"Mean Absolute Diff: {mean_diff:.2e}")
    print(f"Max Absolute Diff:  {max_diff:.2e}")
    print(f"Rel. Norm Diff:     {norm_diff:.2e}")
    
    print("\n--- Debug Voxel 0 ---")
    print("CVXPY (first 5):", X_cvxpy[:, 0][:5])
    print("SCICO (first 5):", np.array(X_scico)[:, 0][:5])
    print("Difference:     ", X_cvxpy[:, 0][:5] - np.array(X_scico)[:, 0][:5])

    # Sometimes ECOS vs ADMM tolerance gives differences around 1e-3 or 1e-4.
    if norm_diff < 1e-2:
        print("Accuracy: Matches (within expected tolerance).")
    else:
        print("Accuracy: MISMATCH likely (check regularization formulation).")

if __name__ == "__main__":
    benchmark_inverse_solver_contest()
