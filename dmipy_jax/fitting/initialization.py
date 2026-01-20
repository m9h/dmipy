
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Tuple, List, Callable, Any

class GlobalBruteInitializer:
    """
    A Global Grid Search initializer for non-linear least squares fitting.
    
    This replaces the "Brute" phase of the legacy Brute2FineOptimizer.
    It evaluates the model on a grid of parameters and selects the best starting point
    for the gradient-based optimizer.
    """
    
    def __init__(self, model, n_grid_points: int = 5):
        """
        Args:
            model: A model instance (e.g. JaxMultiCompartmentModel) or callable.
                   Must operate on "unscaled" (physical) parameters if accessing ranges directly,
                   but the generic initializer works with ranges provided.
                   Here we assume it works with the `JaxMultiCompartmentModel` interface.
            n_grid_points: Number of points per dimension for the grid. 
                           BEWARE: Complexity is N^D. Keep small (3-5).
        """
        self.model = model
        self.n_grid_points = n_grid_points

    def _generate_grid(self, parameter_ranges: Dict[str, Any], parameter_cardinality: Dict[str, int]):
        """
        Generates a meshgrid of parameters.
        
        Returns:
            jnp.ndarray: shape (N_combinations, N_total_params)
        """
        axes = []
        
        # Iterate in the order of model parameters
        # JaxMultiCompartmentModel exposes parameter_names in order corresponding to the flat array
        
        if not hasattr(self.model, 'parameter_names'):
             raise ValueError("Model must have 'parameter_names' for automatic grid generation.")

        for name in self.model.parameter_names:
            card = parameter_cardinality[name]
            rng = parameter_ranges[name] # tuple (min, max) or list of tuples
            
            if card == 1:
                # scalar
                low, high = rng
                if jnp.isinf(low) or jnp.isinf(high):
                    # Fallback for unbounded: assume 0-1 or some heuristic if name implies something?
                    # For now, default to 0.1-2.0 roughly? Or raise error.
                    # Legacy dmipy requires bounds for Brute2Fine.
                    if "mu" in name: # Orientation usually handled differently (sphere grid)
                         # Skipping orientation grid search typicaly requires spherical grid
                         # For now, simplistic linspace invalid for sphere.
                         low, high = 0.0, jnp.pi # Unsafe placeholder for spherical coords
                    else:
                         low, high = 0.1, 1.0 # Very risky default
                
                # Create grid points
                axis = jnp.linspace(low, high, self.n_grid_points)
                axes.append(axis)
            else:
                # Vector parameter
                # rng should be list of (min, max)
                # Iterate each dimension
                if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], (int, float)):
                    # Broadcast range
                    for _ in range(card):
                        low, high = rng
                        axis = jnp.linspace(low, high, self.n_grid_points)
                        axes.append(axis)
                else:
                    for i in range(card):
                        low, high = rng[i]
                        axis = jnp.linspace(low, high, self.n_grid_points)
                        axes.append(axis)
                        
        # Meshgrid
        # This explodes quickly. 
        # Legacy dmipy often uses "Cascade" or tailored grids.
        # Global Brute on 10 parameters with 5 points = 5^10 ~ 9 million evaluations per voxel. Too slow.
        # We need a smarter approach or user-provided grid.
        
        # For this implementation, we will assume reasonable dimensionality or user-provided random sampling?
        # Let's switch to Random Sampling (Random Search) for high dimensions if N > 4
        # Or simple meshgrid.
        
        mesh = jnp.meshgrid(*axes, indexing='ij')
        
        # Flatten
        # stack -> (N_params, D1, D2, ...) -> flatten last dims
        flat_mesh = jnp.stack([m.ravel() for m in mesh], axis=-1)
        return flat_mesh

    @partial(jax.jit, static_argnums=(0,))
    def compute_initial_guess(self, data, acquisition, init_grid):
        """
        Selects the best parameter set from the grid for the given data.
        
        Args:
            data: Signal data (N_meas,)
            acquisition: JaxAcquisition
            init_grid: (N_candidates, N_params) array of parameter candidates.
            
        Returns:
            best_params: (N_params,)
        """
        
        # Define evaluation function for one set of params
        def evaluate_one(params):
            # model_func signature: (params, acquisition) -> signal
            # attributes of self.model might change, but self.model.model_func is the curve function
            pred = self.model.model_func(params, acquisition)
            mse = jnp.mean((data - pred)**2)
            return mse
            
        # Vmap over candidates
        mse_scores = jax.vmap(evaluate_one)(init_grid)
        
        # Find argmin
        best_idx = jnp.argmin(mse_scores)
        return init_grid[best_idx]

    def generate_random_grid(self, n_samples=1000, key=None):
        """Generates random parameter candidates within bounds."""
        if key is None:
            key = jax.random.PRNGKey(42)
            
        params = []
        keys = jax.random.split(key, len(self.model.parameter_names))
        
        flat_ranges = []
        # Re-construct flat ranges list similar to modeling_framework
        # This duplication logic suggests `JaxMultiCompartmentModel` should expose `flat_ranges`.
        # Assuming we can access `self.model.parameter_ranges` etc.
        
        # Simplified: We assume we can get a list of low/high.
        # Let's iterate.
        
        dim_idx = 0
        grid_cols = []
        
        # We need to flatten the range structure first
        # TO BE IMPLEMENTED: Robust range flattening
        # ideally JaxMultiCompartmentModel should provide `get_flat_bounds()`
        
        # Placeholder return for now
        return None 
