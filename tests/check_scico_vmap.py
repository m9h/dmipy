
import jax
import jax.numpy as jnp
from scico import functional, linop, loss, optimize
import numpy as np

def test_vmap_admm():
    # Problem dimensions
    M = 10 # measurements
    N = 5  # unknowns
    Batch = 3 # batch size

    A = jnp.ones((M, N))
    y_batch = jnp.ones((Batch, M))

    def solve_single(y):
        # A simple non-negative least squares: min ||y - Ax||^2 s.t. x>=0
        f = loss.SquaredL2Loss(y=y, A=linop.MatrixOperator(A))
        g = functional.NonNegativeIndicator()
        
        solver = optimize.ADMM(
            f=f,
            g_list=[g],
            C_list=[linop.Identity(N)],
            rho_list=[1.0],
            x0=jnp.zeros(N),
            maxiter=10
        )
        return solver.solve()

    print("Attempting to vmap scico.optimize.ADMM...")
    try:
        x_hat = jax.vmap(solve_single)(y_batch)
        print("Success! x_hat shape:", x_hat.shape)
    except Exception as e:
        print("Caught expected error:")
        print(e)

if __name__ == "__main__":
    test_vmap_admm()
