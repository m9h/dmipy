
import jax
import jax.numpy as jnp
from jax import lax
import lineax as lx

class BatchedLevenbergMarquardt:
    """
    High-performance Levenberg-Marquardt solver optimized for 
    batch processing of small, dense problems on GPU.
    
    Structure:
    - Fixed number of iterations (unrolled via lax.scan for max throughput).
    - Uses Cholesky factorization for the normal equations.
    - Simple damping heuristic (Marquardt).
    """
    
    def __init__(self, max_steps=20, damping=1e-3):
        self.max_steps = max_steps
        self.damping = damping

    def solve(self, residual_fun, init_params, args=()):
        """
        Solves the non-linear least squares problem.
        
        Args:
            residual_fun: Function (params, args) -> residuals (1D array)
            init_params: Initial guess (1D array)
            args: Extra arguments for residual_fun
            
        Returns:
            final_params: Optimized parameters
            final_stats: Dictionary with step counts (fixed) and final cost.
        """
        
        # 1. Setup State
        # State: (params, lambda, cost)
        # We need to compute initial cost
        init_r = residual_fun(init_params, args)
        init_cost = 0.5 * jnp.sum(init_r**2)
        
        init_state = (init_params, self.damping, init_cost)
        
        # 2. Define Step Function
        def step_fn(state, _):
            params, lam, cost = state
            
            # Linearize: r(p) + J*delta approx 0
            # J*delta = -r
            # Normal Eq: (J^T J + lam*I) delta = -J^T r
            
            # Use AD to get J and r
            r = residual_fun(params, args)
            J = jax.jacfwd(residual_fun)(params, args)
            
            # Shapes: r [M], J [M, N]
            # J^T J [N, N]
            JtJ = jnp.matmul(J.T, J)
            Jtr = jnp.matmul(J.T, r)
            
            # Damping
            N = JtJ.shape[0]
            # Scale invariant damping? Or simple additive?
            # Marquardt: diag(JtJ) + lam * I? Or just JtJ + lam * diag(JtJ)?
            # Let's use simple Levenberg for now: JtJ + lam * I
            # Or Levenberg-Marquardt: JtJ + lam * diag(JtJ)
             
            # Robust version:
            # If diag contains zeros, add epsilon?
            diag_JtJ = jnp.diag(JtJ)
            damping_matrix =  (lam * jnp.eye(N))
            # Or better: (JtJ + lam * diag(JtJ))
            # LHS = JtJ + lam * jnp.diag(diag_JtJ) # This is LM
            # Let's stick to Levenberg (lam * I) for stability first, easier to implement
            
            # Check if Cholesky works, if not increase lam?
            # Hard to do conditional control flow for "try-fail-update" inside scan efficiently without while_loop.
            
            # For high-throughput fixed-step, we use a Trust Region-like update or simple step acceptance.
            
            LHS = JtJ + lam * jnp.eye(N)
            RHS = -Jtr
            
            # Solve
            # delta = jnp.linalg.solve(LHS, RHS) 
            # Use Cholesky for speed on small matrices
            # If LHS not pos-def (due to num errors), this might nan. 
            # Add jitter?
            LHS_jit = LHS + 1e-6 * jnp.eye(N)
            
            # delta = jax.scipy.linalg.solve(LHS_jit, RHS, assume_a='pos')
            # Using Lineax explicitly
            op = lx.MatrixLinearOperator(LHS_jit)
            # Use AutoLinearSolver with well_posed=False to handle singular/ill-conditioned matrices
            solver = lx.AutoLinearSolver(well_posed=False)
            delta = lx.linear_solve(op, RHS, solver=solver).value
            
            # Candidate
            new_params = params + delta
            
            # Eval new cost
            new_r = residual_fun(new_params, args)
            new_cost = 0.5 * jnp.sum(new_r**2)
            
            # Acceptance Step
            # If cost decreases, accept and decrease lam
            # If cost increases, reject and increase lam
            
            accept = new_cost < cost
            
            next_params = jnp.where(accept, new_params, params)
            next_cost = jnp.where(accept, new_cost, cost)
            next_lam = jnp.where(accept, lam * 0.1, lam * 10.0)
            
            # Clamp lam to avoid explosion/underflow
            next_lam = jnp.clip(next_lam, 1e-9, 1e9)
            
            return (next_params, next_lam, next_cost), None

        # 3. Run Loop
        # lax.scan is unrollable
        final_state, _ = lax.scan(step_fn, init_state, None, length=self.max_steps)
        
        final_params, _, final_cost = final_state
        
        stats = {
            'final_cost': final_cost,
            'steps': self.max_steps # Fixed
        }
        
        return final_params, stats


class BatchedProximalLevenbergMarquardt:
    """
    Proximal Levenberg-Marquardt solver for L1-regularized problems on GPU.
    Optimizes: 0.5 * ||f(x) - y||^2 + g(x)
    where g(x) is a convex function with a known proximal operator (e.g. L1 norm).

    Algorithm:
    Iterative step: x_{k+1} = prox_{lambda*step}( x_k - (J^T J + mu I)^-1 J^T r )
    NOTE: Standard Proximal Gradient is x - learning_rate * grad. 
    Proximal Newton/LM is more complex:
    x_{k+1} = prox_{scaled}( x_k - H^-1 g ) ?
    Actually, for L1, we usually use FISTA for simple problems or Proximal Newton.
    
    Simplified Approach (Proximal Gradient with Hessian approximation?):
    This implementation applies the proximal operator AFTER the Levenberg-Marquardt update.
    This is an approximation but works well for sparsity inducing in practice if steps are small.
    
    A more rigorous approach is to solve the subproblem:
    minDelta 0.5 ||r + J Delta||^2 + 0.5 mu ||Delta||^2 + g(x + Delta)
    
    For now, we implement the "Post-Update Proximal" heuristic:
    1. Compute LM update Delta (ignoring g(x))
    2. x_temp = x + Delta
    3. x_new = prox(x_temp)
    """
    
    def __init__(self, max_steps=20, damping=1e-3):
        self.max_steps = max_steps
        self.damping = damping

    def solve(self, residual_fun, init_params, prox_op, args=()):
        """
        Solves the proximal non-linear least squares problem.
        
        Args:
            residual_fun: Function (params, args) -> residuals
            init_params: Initial guesses
            prox_op: Function (params) -> params (applies proximal operator)
            args: Extra args
        """
        
        # 1. Init State
        init_r = residual_fun(init_params, args)
        init_cost = 0.5 * jnp.sum(init_r**2)
        # Note: Cost tracking here only tracks MSE, not full composite objective
        # To decide acceptance properly we should track composite cost?
        # Typically LM uses MSE for trust region.
        
        init_state = (init_params, self.damping, init_cost)
        
        def step_fn(state, _):
            params, lam, cost = state
            
            # Linearize and compute Update
            r = residual_fun(params, args)
            J = jax.jacfwd(residual_fun)(params, args)
            
            JtJ = jnp.matmul(J.T, J)
            Jtr = jnp.matmul(J.T, r)
            
            N = JtJ.shape[0]
            
            # Levenberg-Marquardt Step
            LHS = JtJ + lam * jnp.eye(N)
            LHS_jit = LHS + 1e-6 * jnp.eye(N)
            RHS = -Jtr
            
            # Solve for delta
            # delta = (J^T J + lam I)^-1 (-J^T r)
            op = lx.MatrixLinearOperator(LHS_jit)
            solver = lx.AutoLinearSolver(well_posed=False)
            delta = lx.linear_solve(op, RHS, solver=solver).value
            
            # Candidate before prox
            cond_params = params + delta
            
            # Apply Proximal Operator
            # x_{k+1} = prox(x_k + delta)
            # Note: ideally prox parameter depends on Hessian scaling, difficult in batched LM.
            # We assume constant lambda embedded in prox_op
            new_params = prox_op(cond_params)
            
            # Eval new cost (MSE only or Composite?)
            # Standard LM checks MSE reduction.
            # If we enforce sparsity, MSE might go UP slightly but objective goes down?
            # Let's check MSE for stability.
            new_r = residual_fun(new_params, args)
            new_cost = 0.5 * jnp.sum(new_r**2)
            
            # Acceptance
            accept = new_cost < cost
            
            next_params = jnp.where(accept, new_params, params)
            next_cost = jnp.where(accept, new_cost, cost)
            next_lam = jnp.where(accept, lam * 0.1, lam * 10.0)
            next_lam = jnp.clip(next_lam, 1e-9, 1e9)
            
            return (next_params, next_lam, next_cost), None
            
        final_state, _ = lax.scan(step_fn, init_state, None, length=self.max_steps)
        final_params, _, final_cost = final_state
        
        stats = {'final_cost': final_cost, 'steps': self.max_steps}
        return final_params, stats

