
import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from typing import Callable, Optional
from .buckling import BucklingSimulator
from .metrics import compute_stress_fa

class InverseSolver:
    """
    Solves the Microstructure-to-Stress Inverse Problem.
    
    Minimize L(growth_params) = || FA_target - StressFA(Simulator(growth_params)) ||^2
    """
    
    def __init__(self, simulator: BucklingSimulator, parameterizer: Optional[Callable] = None):
        self.simulator = simulator
        self.nn, self.ne = simulator.get_mesh_info()
        self.parameterizer = parameterizer
        
    def _default_parameterizer(self, params: np.ndarray) -> np.ndarray:
        """
        Default: assume params IS the growth map (Identity).
        If params is scalar, uniform growth.
        """
        if params.size == 1:
            return np.full(self.nn, params.item())
        elif params.size == self.nn:
            return params
        else:
            # Simple interpolation or tiling?
            # For now raise error
            raise ValueError(f"Params size {params.size} does not match nodes {self.nn}")

    def solve(self, target_fa: np.ndarray, initial_guess: np.ndarray, 
              max_iter: int = 50, method: str = 'Powell'):
        """
        Runs the optimization loop.
        
        Args:
            target_fa: (N_elements,) array of target FA values (derived from patient dMRI).
                       Note: Simulator outputs element-wise stress, so we compare element-wise FA.
            initial_guess: Initial parameters for growth map.
            
        Returns:
            result: scipy.optimize.OptimizeResult
        """
        param_fn = self.parameterizer if self.parameterizer else self._default_parameterizer
        
        def objective(params):
            # 1. Decode params to full growth map
            growth_map = param_fn(params)
            
            # 2. Run Forward Model (Black Box)
            # Use try-catch to avoid crashing optimizer on unstable params
            try:
                stress_tensors = self.simulator.run_simulation(growth_map)
            except RuntimeError as e:
                print(f"Simulation failed: {e}")
                return 1e9 # Large penalty
            
            # 3. Compute Metric
            # stress_tensors: (Ne, 3, 3)
            # Use JAX for speed/correctness
            sim_fa = compute_stress_fa(jnp.asarray(stress_tensors)) # (Ne,)
            sim_fa_np = np.array(sim_fa)
            
            # 4. Loss (MSE)
            loss = np.mean((target_fa - sim_fa_np)**2)
            
            print(f"Loss: {loss:.6f}")
            return loss

        print(f"Starting Optimization with method {method}...")
        res = scipy.optimize.minimize(
            objective, 
            initial_guess, 
            method=method,
            options={'maxiter': max_iter, 'disp': True}
        )
        
        return res
