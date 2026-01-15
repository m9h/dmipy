import time
import numpy as np
import scipy.optimize
import jax
import jax.numpy as jnp
import jaxopt
from jax import jit, vmap

# ---------------------------------------------------------------------------
# 1. Setup & Constants
# ---------------------------------------------------------------------------

N_VOXELS_TOTAL = 1_000_000
N_DIRECTIONS = 96
N_WARMUP = 100
N_LEGACY = 100

# Constants for the acquisition
BVALS_SHELLS = np.array([1000, 2000, 3000])
# Create simple multi-shell scheme: 32 dirs per shell
np.random.seed(42)
def fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

gradient_directions = fibonacci_sphere(N_DIRECTIONS)
# Assign b-values (approximate shells)
bvals = np.concatenate([
    np.full(32, 1000.0),
    np.full(32, 2000.0),
    np.full(32, 3000.0)
])
# Ensure shapes match just in case
if len(bvals) != len(gradient_directions):
    # Adjust if N_DIRECTIONS != 96 exactly in my manual setup, but here it matches 32*3=96
    pass

# Diffusion Constants
D_PAR = 1.7e-9 * 1e6  # um^2/ms approx, scaling for simple numbers. Let's use SI or standard units.
# Usually b=1000 s/mm^2. D ~ 0.7 - 2.0 x 10^-3 mm^2/s = um^2/ms.
# Let's work in s/mm^2 for b and mm^2/s for D.
# b=1000 - 3000 s/mm^2
# D_intra = 1.7e-3 mm^2/s
D_INTRA = 1.7e-3 
D_ISO = 3.0e-3 

# Integration Grid for "Heavyweight" Dispersion (Watson)
# We integrate over a sphere to simulate dispersion around the main axis [1,0,0]
N_INTEGRATION_GRID = 50 
INTEGRATION_GRID = fibonacci_sphere(N_INTEGRATION_GRID)

# ---------------------------------------------------------------------------
# 2. Model Definitions
# ---------------------------------------------------------------------------

def watson_weight_numpy(kappa, mu, grid):
    # W(n) = exp(kappa * (n.mu)^2) / ConfluentHyperconf...
    # For benchmark, we ignore the normalizing constant's complexity or approximate it? 
    # Normalizing constant C(k) is important for gradients. 
    # M(1/2, 3/2, k) 
    # We will use the unnormalized weight and normalize numerically for "Heavyweight" feel.
    # weights = exp(k * (n.mu)^2)
    # norm = sum(weights)
    dot_prod = np.dot(grid, mu)
    weights = np.exp(kappa * dot_prod**2)
    return weights / np.sum(weights)

def watson_weight_jax(kappa, mu, grid):
    dot_prod = jnp.dot(grid, mu)
    weights = jnp.exp(kappa * dot_prod**2)
    return weights / jnp.sum(weights)

def signal_model_numpy(params, bvals, bvecs, grid, mu_fiber):
    # params: [f_ic, kappa, f_iso]
    f_ic, kappa, f_iso = params
    
    # 1. Isotropic Signal
    # E_iso = exp(-b * D_iso)
    E_iso = np.exp(-bvals * D_ISO)
    
    # 2. Anisotropic Signal (Intra + Extra)
    # We compute the signal for current bvecs assuming a single stick/zeppelin aligned with grid vectors,
    # then optimally combine them.
    
    # Pre-calculate signal for all grid orientations? 
    # No, that's too much memory. We loop or broadcast.
    # For numpy "legacy", we might implement it naively (slow).
    
    # Weights for the grid based on current kappa
    weights = watson_weight_numpy(kappa, mu_fiber, grid) # (N_grid,)
    
    # Intra (Stick) along grid direction n: exp(-b * D_par * (g.n)^2)
    # Extra (Zeppelin) along n: exp(-b * [ D_par(g.n)^2 + D_perp(1-(g.n)^2) ])
    # Tortuosity: D_perp = D_par * (1 - f_ic)
    D_perp = D_INTRA * (1.0 - f_ic)
    
    # Micro-env signal E_micro = f_ic * E_stick + (1-f_ic) * E_zep
    
    # This loop over grid is the "Heavyweight" part
    E_micro_averaged = np.zeros_like(bvals)
    
    for i, n in enumerate(grid):
        gn = np.dot(bvecs, n) # (N_meas,)
        gn2 = gn**2
        
        E_stick = np.exp(-bvals * D_INTRA * gn2)
        E_zep = np.exp(-bvals * (D_INTRA * gn2 + D_perp * (1 - gn2)))
        
        E_unweighted = f_ic * E_stick + (1.0 - f_ic) * E_zep
        E_micro_averaged += weights[i] * E_unweighted
        
    # Total Signal
    S = f_iso * E_iso + (1.0 - f_iso) * E_micro_averaged
    return S

def signal_model_jax(params, bvals, bvecs, grid, mu_fiber):
    f_ic, kappa, f_iso = params
    
    E_iso = jnp.exp(-bvals * D_ISO)
    
    weights = watson_weight_jax(kappa, mu_fiber, grid) # (N_grid,)
    
    D_perp = D_INTRA * (1.0 - f_ic)
    
    # Vectorized grid calc
    # grid: (N_grid, 3)
    # bvecs: (N_meas, 3)
    # gn: (N_meas, N_grid)
    gn = jnp.dot(bvecs, grid.T)
    gn2 = gn**2
    
    E_stick = jnp.exp(-bvals[:, None] * D_INTRA * gn2)
    # E_zep: exp of ...
    # D_par * gn2 + D_perp * (1-gn2)
    exponent = D_INTRA * gn2 + D_perp * (1.0 - gn2)
    E_zep = jnp.exp(-bvals[:, None] * exponent)
    
    # E_micro for each grid point
    E_micro_grid = f_ic * E_stick + (1.0 - f_ic) * E_zep # (N_meas, N_grid)
    
    # Average
    # weights: (N_grid,)
    E_micro_averaged = jnp.dot(E_micro_grid, weights)
    
    S = f_iso * E_iso + (1.0 - f_iso) * E_micro_averaged
    return S

# ---------------------------------------------------------------------------
# 3. Data Generation
# ---------------------------------------------------------------------------

print(f"Generating synthetic data for {N_VOXELS_TOTAL} voxels (Fast Random)...")
# Generate random signal data [0, 1] for benchmarking optimizer speed.
# We don't need perfect biophysics for the speed test.

mu_fiber_fixed = np.array([1.0, 0.0, 0.0]) # Fixed fiber direction for fitting

data_noisy_np = np.random.rand(N_VOXELS_TOTAL, N_DIRECTIONS).astype(np.float32)
data_noisy_jax = jnp.array(data_noisy_np)
print("Data generated.")

# ---------------------------------------------------------------------------
# 4. Fitting Functions
# ---------------------------------------------------------------------------

# Bounds: f_ic [0,1], kappa [0, 32], f_iso [0,1]
BOUNDS_MIN = [0.01, 0.01, 0.0]
BOUNDS_MAX = [0.99, 32.0, 0.99]
X0 = [0.5, 4.0, 0.1] # Initial guess

def fit_legacy_scipy(n_voxels):
    # Subset
    subset_data = data_noisy_np[:n_voxels]
    
    results = []
    
    start_time = time.time()
    for i in range(n_voxels):
        vox_data = subset_data[i]
        
        def objective(params):
            # params bounds check normally done by optimizer, 
            # here just compute SSE
            pred = signal_model_numpy(params, bvals, gradient_directions, INTEGRATION_GRID, mu_fiber_fixed)
            return np.sum((pred - vox_data)**2)
            
        res = scipy.optimize.minimize(objective, X0, bounds=list(zip(BOUNDS_MIN, BOUNDS_MAX)), method='L-BFGS-B')
        results.append(res.x)
        
    end_time = time.time()
    return end_time - start_time

def fit_jax_vmap(n_voxels, data_subset=None):
    if data_subset is None:
        # For the full run
        data_subset = data_noisy_jax[:n_voxels]
    
    # Constants on device
    bvals_j = jnp.array(bvals)
    bvecs_j = jnp.array(gradient_directions)
    grid_j = jnp.array(INTEGRATION_GRID)
    mu_j = jnp.array(mu_fiber_fixed)
    bounds_min_j = jnp.array(BOUNDS_MIN)
    bounds_max_j = jnp.array(BOUNDS_MAX)
    init_params_j = jnp.array(X0)
    
    def objective(params, data_voxel):
        pred = signal_model_jax(params, bvals_j, bvecs_j, grid_j, mu_j)
        return jnp.sum((pred - data_voxel)**2)
    
    # Solver - Use pure JAX LBFGSB
    solver = jaxopt.LBFGSB(fun=objective, tol=1e-3, maxiter=30)
    
    def run_solver(data_v):
        # bounds argument handling depends on jaxopt version, usually passed to run
        return solver.run(init_params_j, bounds=(bounds_min_j, bounds_max_j), data_voxel=data_v)
        
    # vmap
    vmapped_solver = vmap(run_solver)
    
    # JIT
    jf = jit(vmapped_solver)
    
    # Run
    # Block until ready for accurate timing
    res = jf(data_subset)
    res.params.block_until_ready()
    return res

# ---------------------------------------------------------------------------
# 5. Benchmark Execution
# ---------------------------------------------------------------------------

print("\n--- Starting Legacy Benchmark (Scipy) ---")
print(f"Running on {N_LEGACY} voxels...")
legacy_duration = fit_legacy_scipy(N_LEGACY)
legacy_estimated_total = (legacy_duration / N_LEGACY) * N_VOXELS_TOTAL
legacy_hours = legacy_estimated_total / 3600.0
print(f"Legacy Time ({N_LEGACY} voxels): {legacy_duration:.4f} s")
print(f"Legacy Estimated Time (1M voxels): {legacy_hours:.2f} hours")

print("\n--- Starting JAX Benchmark (vmap GPU/TPU/CPU) ---")
# Warmup
print(f"Warmup JIT ({N_WARMUP} voxels)...")
start_warm = time.perf_counter()
fit_jax_vmap(N_WARMUP)
end_warm = time.perf_counter()
print(f"Warmup Complete: {end_warm - start_warm:.4f} s")

# Full Run
print(f"Running on FULL {N_VOXELS_TOTAL} voxels...")
start_jax = time.perf_counter()
# We use the full dataset
# Note: If memory is an issue on the specific device, we might batch, 
# but vmap usually handles 1M scalars fine if model isn't huge. 
# 1M * 96 * float32 ~ 400MB data. Parameters: small. 
# Intermediate activations in L-BFGS might be large.
try:
    fit_jax_vmap(N_VOXELS_TOTAL, data_noisy_jax)
    end_jax = time.perf_counter()
    jax_duration = end_jax - start_jax
    jax_minutes = jax_duration / 60.0
    
    print("\n---------------------------------------------------")
    print(f"{'Method':<20} | {'Time':<15} | {'Speedup':<10}")
    print("---------------------------------------------------")
    print(f"{'Legacy (Est)':<20} | {legacy_hours:.4f} h      | 1.0x")
    actual_time_str = f"{jax_minutes:.4f} m"
    speedup = legacy_estimated_total / jax_duration
    print(f"{'JAX (Actual)':<20} | {actual_time_str:<15} | {speedup:.1f}x")
    print("---------------------------------------------------")
    
except Exception as e:
    print(f"\nJAX Run Failed (possibly OOM on 1M voxels?): {e}")
    print("Try reducing N_VOXELS_TOTAL if this happens.")
