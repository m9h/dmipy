import jax
import jax.numpy as jnp
import jaxopt
from functools import partial

__all__ = ['VoxelFitter', 'mse_loss']

def mse_loss(params, data, bvals, bvecs, model_func, unwrap_fn):
    args = unwrap_fn(params)
    prediction = model_func(bvals, bvecs, *args)
    return jnp.mean((data - prediction) ** 2)

class VoxelFitter:
    """
    A JAX-optimized fitter using the Pure-JAX L-BFGS-B solver.
    This allows full compilation and vectorization on the GPU.
    """
    def __init__(self, model_func, parameter_ranges):
        self.model_func = model_func
        # Convert list of (min, max) into tuple of arrays for JAX
        # Format for LBFGSB is (lower_array, upper_array)
        lower_bounds, upper_bounds = zip(*parameter_ranges)
        self.bounds = (jnp.array(lower_bounds), jnp.array(upper_bounds))

    # Static argnums tells JAX to recompile if 'self' changes (which it won't here)
    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data, bvals, bvecs, init_params):
        """
        Fits a single voxel. Can be vmapped over 'data' and 'init_params'.
        """
        
        # 1. Define the unpacker
        def unwrap_identity(p):
            return tuple(p)

        # 2. Define the objective closure
        def objective(params, data, bvals, bvecs):
            args = unwrap_identity(params)
            prediction = self.model_func(bvals, bvecs, *args)
            return jnp.mean((data - prediction) ** 2)

        # 3. Instantiate the Pure JAX Solver
        # LBFGSB is written in JAX, so it compiles perfectly on GPU
        solver = jaxopt.LBFGSB(
            fun=objective,
            maxiter=20,     # Cap iterations for speed
            tol=1e-4        # Standard convergence tolerance
        )

        # 4. Run
        sol = solver.run(
            init_params, 
            bounds=self.bounds,
            data=data,
            bvals=bvals, 
            bvecs=bvecs
        )
        return sol.params, sol.state
