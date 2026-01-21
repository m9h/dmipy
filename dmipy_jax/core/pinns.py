
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from typing import Union, Tuple, Optional
from functools import partial

from dmipy_jax.core.networks import SIREN
from dmipy_jax.core.integration_grids import get_spherical_fibonacci_grid
from dmipy_jax.core.loss import rician_nll

class SIREN_CSD(eqx.Module):
    """
    Continuous Spherical Deconvolution using a SIREN (Sinusoidal Representation Network)
    to represent the Fiber Orientation Distribution (FOD).
    
    The signal is computed via direct numerical integration on a sphere:
    S(b) = Integral { FOD(n) * Response(b, n) } dn
         ~ Sum { w_i * FOD(n_i) * Response(b, n_i) }
    """
    siren: SIREN
    response_evals: jnp.ndarray
    grid_points: jnp.ndarray
    grid_weights: jnp.ndarray
    sigma: float
    
    def __init__(
        self,
        response_evals: Union[jnp.ndarray, Tuple[float, float, float]],
        key: jax.Array,
        sigma: float = 0.05,
        n_integration_points: int = 1000,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        omega_0: float = 30.0
    ):
        """
        Args:
            response_evals: Eigenvalues of the single-fiber response function (eval1, eval2, eval3).
            key: JAX PRNG key for initialization.
            sigma: Rician noise level estimate for loss.
            n_integration_points: Number of points for spherical integration grid.
            hidden_features: Width of SIREN hidden layers.
            hidden_layers: Number of SIREN hidden layers.
            omega_0: Frequency scaling factor for SIREN.
        """
        self.response_evals = jnp.array(response_evals) if not isinstance(response_evals, jnp.ndarray) else response_evals
        self.sigma = sigma
        
        # Initialize Integration Grid
        self.grid_points, self.grid_weights = get_spherical_fibonacci_grid(n_integration_points)
        
        # Initialize SIREN
        # Input: (x, y, z) -> 3
        # Output: FOD amplitude -> 1
        self.siren = SIREN(
            in_features=3,
            out_features=1,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            key=key,
            first_omega_0=omega_0,
            hidden_omega_0=omega_0
        )
        
    def _response_kernel(self, bvecs, bval):
        """
        Computes the response kernel matrix K[i, j] = R(bvecs[i], grid_points[j])
        
        R(b, n) = exp( -bval * (lambda1 * (b.n)^2 + lambda2 * (1 - (b.n)^2)) )
        """
        # (N_meas, 3) @ (N_grid, 3).T -> (N_meas, N_grid) of cosines
        cos_theta = jnp.dot(bvecs, self.grid_points.T)
        cos_sq = cos_theta ** 2
        sin_sq = 1.0 - cos_sq
        
        lambda1 = self.response_evals[0]
        lambda2 = self.response_evals[1]
        
        # Assuming lambda2 approx lambda3 for stick/cylinder
        exponent = -bval * (lambda1 * cos_sq + lambda2 * sin_sq)
        return jnp.exp(exponent)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Directly queries the SIREN at spatial coordinate x.
        Note: This returns the raw output. For FOD, use softplus.
        """
        return self.siren(x)

    def get_fod(self, points: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates the FOD at given points on the sphere.
        Enforces positivity via softplus.
        """
        # vmap over points: (N, 3) -> (N, 1)
        # SIREN expects single input (3,)
        pixel_fn = jax.vmap(self.siren)
        raw_out = pixel_fn(points)
        return jax.nn.softplus(raw_out).squeeze()

    def predict_signal(self, bvecs: jnp.ndarray, bval: float) -> jnp.ndarray:
        """
        Predicts signal for a given shell.
        """
        # 1. Evaluate FOD at all grid points
        fod_values = self.get_fod(self.grid_points) # (N_grid,)
        
        # 2. Compute Integration Weights
        # Integral ~ Sum(w_i * f_i)
        weighted_fod = fod_values * self.grid_weights
        
        # 3. Compute Kernel
        kernel = self._response_kernel(bvecs, bval) # (N_meas, N_grid)
        
        # 4. Integrate
        # Signal = Kernel @ weighted_FOD
        return kernel @ weighted_fod

    def fit_voxel(self, data: jnp.ndarray, bvecs: jnp.ndarray, bval: float):
        """
        Fits the SIREN CSD model to a single voxel's data.
        Returns the trained model and the loss.
        """
        
        def loss_fn(model, args):
            d_obs, bvecs_in, bval_in = args
            
            # Predict
            pred_signal = model.predict_signal(bvecs_in, bval_in)
            
            # Loss (Rician NLL)
            return rician_nll(d_obs, pred_signal, self.sigma)
            
        solver = optx.BFGS(rtol=1e-3, atol=1e-3)
        
        # Optimistix minimize used for functional optimization over pytrees (params) defined in 'y0'.
        # Since SIREN is an Equinox module, we can optimize 'self' directly if we treat it as the params.
        
        sol = optx.minimise(
            fn=loss_fn,
            solver=solver,
            y0=self, # The model itself is the parameter tree
            args=(data, bvecs, bval),
            max_steps=200,
            throw=False
        )
        
        # Return the optimized model
        return sol.value, sol.result

