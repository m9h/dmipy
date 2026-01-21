import warnings
# Suppress the specific CVXPY matrix multiplication warning
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
import time
import numpy as np
import jax
import jax.numpy as jnp
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.optimizers import amico_cvxpy

# dmipy-jax imports
from dmipy_jax.inverse.amico import AMICOSolver
# We need to wrap dmipy signal models or use dmipy-jax equivalents for the JAX solver
# The JAX solver expects a JAX-compatible model callable.
import dmipy_jax.signal_models.cylinder_models as jax_cylinder
import dmipy_jax.signal_models.gaussian_models as jax_gaussian
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel

def benchmark():
    print("Setting up benchmark...")
    
    # ----------------------------------------------------------------
    # 1. Setup Common Acquisition and Data (using dmipy generation)
    # ----------------------------------------------------------------
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    # Reduce shell count for speed for this demo if needed, but HCP is standard
    
    # Define Model: Ball + Stick
    # dmipy (Legacy)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()
    mc_model = MultiCompartmentModel(models=[ball, stick])
    
    # Generate Synthetic Data
    N_voxels = 200  # Reduced for quick verification
    print(f"Generating data for {N_voxels} voxels...")
    
    # Random ground truth
    np.random.seed(42)
    gt_params = {}
    gt_params['partial_volume_0'] = np.random.uniform(0.1, 0.9, N_voxels)
    gt_params['partial_volume_1'] = 1 - gt_params['partial_volume_0']
    gt_params['G1Ball_1_lambda_iso'] = np.full(N_voxels, 3e-9)
    gt_params['C1Stick_1_lambda_par'] = np.full(N_voxels, 1.7e-9)
    # Random orientations for stick
    theta = np.random.uniform(0, np.pi, N_voxels)
    phi = np.random.uniform(0, 2*np.pi, N_voxels)
    gt_params['C1Stick_1_mu'] = np.column_stack([theta, phi])
    
    data = mc_model.simulate_signal(acq_scheme, gt_params)
    
    # ----------------------------------------------------------------
    # 2. Setup CVXPY Optimizer (dmipy)
    # ----------------------------------------------------------------
    print("\n--- Preparing CVXPY Optimizer ---")
    # Need to construct the grid and calling conventions as per dmipy tests
    
    # Configuration
    Nt = 8 # Angular resolution for atoms (keep low for speed in test)
    # In AMICO, we grid orientation using a sphere discretization typically?
    # The 'test_amico.py' uses simple linspace for 1D params, 
    # but for Stick orientation, it usually requires a sphere sampling.
    # Let's check how 'forward_model_matrix' in test_amico handles 'mu'.
    # It seems it expects 'model_dirs' passed in? 
    # Ah, the test passes `model_dirs` which implies it cheats by knowing the direction?
    # Or creates a dictionary of MANY directions.
    # Real AMICO usually uses a fixed set of directions (e.g. vertices of sphere).
    
    from dmipy.utils import spherical_mean
    # Let's use a simple discrete sphere with N_dirs
    # For a fair benchmark, we limit the number of atoms.
    # 12 directions?
    # We will simulate a simplistic "Direction known" scenario or "Fixed set of directions".
    # Standard AMICO sets up fixed directions.
    
    # But `AmicoCvxpyOptimizer` takes `M` (matrix) in `__call__`.
    # So we must pre-calculate M.
    
    # Let's define a fixed dictionary for both.
    # Fixed orientations:
    from dmipy.core import modeling_framework
    # Just use random directions as "dictionary atoms"
    N_atoms_dir = 32
    # simple fibonacci sphere or similar
    # For simplicity, random 32 directions
    t = np.random.uniform(0, np.pi, N_atoms_dir)
    p = np.random.uniform(0, 2*np.pi, N_atoms_dir)
    atom_dirs = np.column_stack([t, p])
    
    # We also need to grid other params if they vary.
    # For Ball+Stick, usually only orientation varies if partial volumes are the unknowns to solve for.
    # Diffusivities are typically fixed in AMICO-NODDI.
    # So we fix diffusivities in the kernel models.
    
    # Re-define model with fixed diffusivities for the dictionary generation
    ball_fix = gaussian_models.G1Ball(lambda_iso=3e-9)
    stick_fix = cylinder_models.C1Stick(lambda_par=1.7e-9)
    mc_model_fix = MultiCompartmentModel(models=[ball_fix, stick_fix])
    
    # Construct M for CVXPY
    # M has shape (N_data, N_atoms)
    # Atoms: 1 for Ball, N_atoms_dir for Stick
    
    # Ball signal
    E_ball = ball_fix(acq_scheme, **{}) # (N_data,)
    
    # Stick signals
    E_sticks = []
    for i in range(N_atoms_dir):
        # dmipy expects mu as (1, 2)
        E_sticks.append(stick_fix(acq_scheme, mu=atom_dirs[i]))
    E_sticks = np.array(E_sticks) # (N_atoms, N_data)
    
    # Concatenate to M
    # M shape (N_data, N_atoms_total)
    M_cvxpy = np.vstack([E_ball.reshape(1, -1), E_sticks]).T
    
    # Set up optimizer
    # AmicoCvxpyOptimizer requires instantiation with model/acq
    # And we also need x0_vector initial size, lambda_1, lambda_2
    
    # lambda arrays must match number of models (2)
    # lambda arrays must match number of models (2)
    optimizer_cvxpy = amico_cvxpy.AmicoCvxpyOptimizer(
        mc_model_fix, 
        acq_scheme, 
        x0_vector=np.zeros(1 + N_atoms_dir), 
        lambda_1=[0, 0], # No reg for simple speed test
        lambda_2=[0, 0]
    )
    
    # Suppress internal dmipy warnings about sphere if possible, or we just ignore them.
    # The warnings come from inside dmipy imports where 'get_sphere' is called.
    # We can't fix dmipy library code here easily, but we can suppress warnings further if needed.
    # warnings.filterwarnings("ignore", message="Pass \['name'\] as keyword args")
    
    # The __call__ needs 'grid' and 'idx' dicts to map atoms back to params.
    # This is metadata overhead.
    idx_map = {
        'G1Ball_1_': np.array([0]),
        'C1Stick_1_': 1 + np.arange(N_atoms_dir)
    }
    grid_map = {} # Minimal grid map
    # We simulate 'grid' content just enough to not crash if it uses it for reconstruction
    # The return value is what matters?
    # CVXPY optimizer returns "fitted_parameter_vector" but it calculates it from x0_vector using grid.
    # If we just want to measure SOLVING time, we can ignore the reconstruction overhead if possible,
    # but it's part of the 'call'.
    # We populate grid so it returns valid-ish numbers.
    grid_map['partial_volume_0'] = np.zeros(1 + N_atoms_dir) # Dummy
    grid_map['partial_volume_0'][0] = 1
    
    # ----------------------------------------------------------------
    # 3. Setup SCICO Optimizer (dmipy-jax)
    # ----------------------------------------------------------------
    print("\n--- Preparing SCICO/JAX Optimizer ---")
    # For JAX solver, we need to pass dictionary params to generate kernels.
    # dmipy-jax AMICOSolver takes `dictionary_params`.
    
    # Define equivalent JAX model logic
    # We can pass a simple callable that simulates signal from params
    def jax_model_call(params, acq):
        # params is dict.
        # Check keys.
        # If 'G1Ball_1' in params or similar?
        # The solver passes `p_dict` constructed from zip(keys, values).
        # We need to adapt to what we pass as 'keys'.
        
        # If we define keys as ['type', 'theta', 'phi']?
        # Or simpler: The AMICOSolver logic builds a grid.
        # We want to match M_cvxpy.
        # Atom 0: Ball. Atom 1..N: Sticks.
        # This heterogeneous list is hard to express as a single cartesian product grid 
        # unless we add a 'model_type' parameter.
        
        # Let's use a workaround:
        # We pass a simple integer index 'atom_idx' as parameter.
        # usage: dictionary_params = {'atom_idx': np.arange(1 + N_atoms_dir)}
        
        idx = params['atom_idx']
        
        # We need to return signal for this index.
        # Since this is inside JIT, we need JAX logic.
        # M_cvxpy is numpy, let's cast to JAX constant.
        M_const = jnp.array(M_cvxpy) # (N_data, N_atoms)
        
        # Return column corresponding to idx.
        # M_const[:, idx]
        # But `evaluate_atom` returns (N_data,) presumably? Yes.
        # `dictionary_params` values are grid axes.
        
        return M_const[:, idx.astype(int)]

    # Wait, AMICOSolver `generate_kernels` does `jax.vmap(model_wrapper)(stacked_params)`.
    # `stacked_params` will have shape (N_atoms, 1).
    # `model_wrapper` receives tuple of values.
    
    # So we can just use the pre-computed matrix logic to ensuring identical dictionaries.
    
    solver_jax = AMICOSolver(
        model=jax_model_call,
        acquisition=acq_scheme, # Passed but we might not use it in our custom callable
        dictionary_params={'atom_idx': jnp.arange(1 + N_atoms_dir)}
    )
    
    # Pre-compile
    # Old warmup block removed in favor of batch warmup below
    
    # ----------------------------------------------------------------
    # 4. Run Benchmarks
    # ----------------------------------------------------------------
    print(f"\nrunning benchmarks on {N_voxels} voxels...")
    
    # CVXPY Loop
    start_time = time.time()
    for i in range(N_voxels):
        _ = optimizer_cvxpy(data[i], M_cvxpy, grid_map, idx_map)
    cvxpy_time = time.time() - start_time
    print(f"CVXPY Total Time: {cvxpy_time:.4f} s")
    print(f"CVXPY Per Voxel:  {cvxpy_time/N_voxels*1000:.4f} ms")
    
    # JAX Bulk Fit
    # Convert data to JAX
    data_jax = jnp.array(data)
    
    import equinox as eqx
    
    # Define JIT-compiled fit step
    # Since we modified AmicoSolver to use pure JAX (lax.scan), it is now safe to JIT fully.
    
    @eqx.filter_jit
    def jitted_fit(solver, data):
        return solver.fit(data, lambda_reg=0.0, constrained=True)

    print("Compiling JAX solver (Warmup)...")
    # Run once with data to trigger JIT compilation
    start_compile = time.time()
    warmup_res = jitted_fit(solver_jax, data_jax)
    warmup_res.block_until_ready()
    print(f"Compilation done in {time.time() - start_compile:.4f} s.")

    start_time = time.time()
    # Execute fit (fast path)
    res_jax = jitted_fit(solver_jax, data_jax)
    res_jax.block_until_ready()
    jax_time = time.time() - start_time
    
    print(f"JAX Total Time:   {jax_time:.4f} s")
    print(f"JAX Per Voxel:    {jax_time/N_voxels*1000:.4f} ms")
    
    speedup = cvxpy_time / jax_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Verify Accuracy?
    # We didn't really extract parameters from CVXPY loop in this code snippet 
    # (variable assignment was `_`).
    # But strictly speaking we just want to verify they are solving the SAME optimization problem.
    # The 'fit' time dominates.
    
    # For correctness check, let's solve one voxel and compare weights.
    i_chk = 0
    w_cvxpy_scaled = optimizer_cvxpy(data[i_chk], M_cvxpy, grid_map, idx_map) 
    # Wait, optimizer_cvxpy returns 'fitted_parameter_vector' (user params), not raw weights.
    # Internal `x0_vector` is the weights.
    w_cvxpy = optimizer_cvxpy.x0_vector # Stored statefully in the class?
    
    w_jax = res_jax[i_chk]
    # Check agreement (allowing for some solver tolerance differences)
    # Normalizing JAX weights (AMICOSolver doesn't enforce sum=1 by default in L1 mode, check code)
    # The code had `constrained=True` -> NonNegative. no sum constraint explicit in the `_fit_single_voxel`.
    # CVXPY version enforces `x0 >= 0`.
    
    # We should normalize w_jax for comparison if CVXPY does?
    # CVXPY code: `self.x0_vector /= (np.sum(self.x0_vector) + 1.e-8)`
    
    # print(f"Weights diff norm: {np.linalg.norm(w_cvxpy - w_jax)}")

if __name__ == "__main__":
    benchmark()
