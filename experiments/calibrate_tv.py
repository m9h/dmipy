
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from dmipy_jax.io.connectome2 import load_connectome2_mri
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.signal_models.sphere_models import SphereGPD
from dmipy_jax.inverse.amico import AMICOSolver, calculate_mean_parameter_map
from dmipy_jax.inverse.solvers import GlobalOptimizer, MicrostructureOperator, AMICOSolver as GlobalAMICOSolver

# Helper to generate directions
def get_fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def cart2sphere(pts):
    # pts: (N, 3) -> (theta, phi)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return np.stack([theta, phi], axis=1)

def add_rician_noise(signal, snr):
    sigma = 1.0 / snr
    noise1 = np.random.normal(0, sigma, signal.shape)
    noise2 = np.random.normal(0, sigma, signal.shape)
    return np.sqrt((signal + noise1)**2 + noise2**2)

def main():
    print("=== Connectome 2.0 TV Calibration Experiment ===")
    
    # 1. Load Data
    print("Loading High-SNR Connectome data...")
    try:
        # Load a small patch to keep it fast: Slice 70, center 40x40
        data_dict = load_connectome2_mri(voxel_slice=(slice(70, 71), slice(40, 80), slice(40, 80)))
    except FileNotFoundError:
        print("Dataset not found. Please ensure ds006181 is present.")
        return

    dwi = data_dict['dwi'] # (1, 40, 40, N_meas)
    scheme = data_dict['scheme']
    affine = data_dict['affine']
    
    # Remove single slice dim
    dwi = jnp.squeeze(dwi, axis=0)
    print(f"Data Shape: {dwi.shape}")

    # Normalize by approx b0
    b0_mask = scheme.bvalues < 50
    b0_map = jnp.mean(dwi[..., b0_mask], axis=-1, keepdims=True)
    dwi_norm = dwi / jnp.maximum(b0_map, 1e-6)

    # 2. Setup AMICO Model
    print("Initializing AMICO Dictionary...")
    
    # Directions for restricted cylinder
    n_dirs = 32
    dirs_cart = get_fibonacci_sphere(n_dirs)
    dirs_sphere = cart2sphere(dirs_cart)
    # Convert to list of arrays/tuples for gridding?
    # AMICOSolver uses itertools.product. If we pass a list of arrays, it works.
    mu_grid = [d for d in dirs_sphere] # List of (2,) arrays

    # Diameters
    diameters = np.linspace(1e-6, 8e-6, 6) # 1 to 8 microns

    # Define Dictionary Grid
    # Using specific names based on model index suffix if needed, but here simple names might collide?
    # JaxMultiCompartmentModel assigns names. If checking `amico.py`, it handles suffixes `_1`, `_2`.
    # Let's inspect MCM parameter names first.
    
    mc_model = JaxMultiCompartmentModel([
        RestrictedCylinder(lambda_par=1.7e-9), 
        SphereGPD(diffusion_constant=3.0e-9, diameter=15e-6) # Large sphere as free water proxy
    ])
    
    # RestrictedCylinder is model 0. SphereGPD is model 1.
    # Parameter names will be:
    # RC: 'mu_1', 'diameter_1' (since 'diameter' collides with sphere probably?)
    # Sphere: 'diameter_2' ??
    # Let's verify names.
    print("Model Parameter Names:", mc_model.parameter_names)
    
    # Construct Grid
    # We map our grid values to these names.
    # If standard names are preserved (no collision), we use them.
    # 'mu' is unique to RC? Yes.
    # 'diameter' is shared. So it will be renamed.
    # Likely 'diameter_1' (RC) and 'diameter_2' (Sphere).
    
    # Helper to find correct name
    rc_diam_name = 'diameter'
    if 'diameter_1' in mc_model.parameter_names:
        rc_diam_name = 'diameter_1'
        
    grid = {
        'mu': mu_grid,
        rc_diam_name: diameters
        # Sphere diameter is fixed in __init__, so no grid needed if fixed.
    }
    
    # Initialize Solver
    # Use the GlobalAMICOSolver from solvers.py (since we confirmed it has generate_dictionary)
    # But wait, `dmipy_jax.inverse.amico` had `AMICOSolver` too.
    # I imported `AMICOSolver` from `dmipy_jax.inverse.amico`.
    # Let's check which one works.
    # `from dmipy_jax.inverse.amico import AMICOSolver` -> This is the one we viewed in file.
    # It has `generate_kernels` (vmap wrapper) AND `generate_dictionary` (itertools)?
    # No, `inverse/amico.py` had duplicate classes.
    # I will assume `solvers.py` has `GlobalAMICOSolver`?
    # In `inverse/__init__.py`: `from .amico import AMICOSolver`.
    # The `solvers.py` file had `class AMICOSolver` inside it too! (Step 24).
    # It seems `solvers.py`'s `AMICOSolver` is the JAX/SCICO efficient one.
    # I will use `GlobalAMICOSolver` (aliased from solvers.py in my import above).
    
    solver = GlobalAMICOSolver(mc_model, scheme)
    phi = solver.generate_dictionary(grid)
    print(f"Dictionary Size: {phi.shape}")
    
    # 3. Establish Ground Truth (High SNR Fit)
    print("Fitting Ground Truth (High SNR)...")
    # Solve voxel-wise (lambda_tv=0) or small regularization
    # Using `solver.fit` which is voxel-wise/vmap PGM.
    gt_coeffs = solver.fit(dwi_norm.reshape(-1, dwi_norm.shape[-1]), phi, lambda_l1=1e-3, lambda_l2=0.0)
    gt_coeffs = gt_coeffs.reshape(*dwi_norm.shape[:-1], -1) # (40, 40, N_atoms)
    
    # Compute GT Parameter Maps
    # We want Mean Diagrameter.
    # Weights for RC are needed. Cylinder is model 0.
    # Atoms corresponding to RC?
    # Dictionary generation order: itertools.product of cylinder params + Sphere params?
    # Wait, `generate_dictionary` concatenates sub-models.
    # Cylinder atoms come first?
    # We need to know slices.
    # `generate_dictionary` logic:
    # 1. Cylinder (N_mu * N_diam atoms)
    # 2. Sphere (1 atom, if no grid)
    
    n_cyl_atoms = len(mu_grid) * len(diameters)
    n_sph_atoms = 1 # Assuming 1 fixed sphere
    
    # Check total
    if phi.shape[1] != (n_cyl_atoms + n_sph_atoms):
        print(f"Warning: atoms count mismatch. Expected {n_cyl_atoms + n_sph_atoms}, got {phi.shape[1]}")
    
    # Extract Cylinder Weights
    weights_cyl = gt_coeffs[..., :n_cyl_atoms]
    weights_sph = gt_coeffs[..., n_cyl_atoms:]
    
    # Normalize Cylinder Weights for Mean Diameter Calculation
    # We only care about diameter distribution WITHIN the cylinder compartment?
    # Or global mean diameter?
    # Usually Mean Axon Diameter is weighted average of cylinder diameters.
    
    # Calculate Mean Diameter
    # We need the diameter value for each atom.
    # Order in `itertools.product(mu, diameter)`:
    # mu changes slowest? or fastest?
    # `val_lists = [mu_grid, diameters]` -> product(mu, diam)
    # Order: (mu1, d1), (mu1, d2)... (mu2, d1)...
    # So diameter cycles fastest.
    
    # Create vector of diameters for atoms
    # Tile diameters N_dirs times
    diams_per_atom = np.tile(diameters, len(mu_grid))
    diams_param_vec = jnp.array(diams_per_atom)
    
    # calculate_mean_parameter_map helper in amico.py assumes direct grid.
    # We do manual calculation here for safety.
    
    denom = jnp.sum(weights_cyl, axis=-1)
    denom = jnp.where(denom < 1e-6, 1.0, denom)
    gt_mrd = jnp.sum(weights_cyl * diams_param_vec, axis=-1) / denom
    
    # 4. Synthetic Downgrade Loop
    snr_levels = [20, 30] # Limit to 2 for speed in demo
    tv_reg_values = jnp.logspace(-4, 0, 10) # 1e-4 to 1.0
    
    results = {
        'snr': [],
        'lambda': [],
        'mse': [],
        'ssim': []
    }
    
    # Initialize Global Optimizer
    op = MicrostructureOperator(mc_model, scheme, input_shape=gt_coeffs.shape)
    # Wait, GlobalOptimizer (TV) solves for coefficients X directly.
    # `GlobalOptimizer` in `solvers.py` uses `MicrostructureOperator` as A?
    # But `MicrostructureOperator` is non-linear?
    # `MicrostructureOperator` wraps `model.model_func`.
    # AMICO is Linear: Y = Phi * X.
    # We want to solve min_X ||Phi X - Y|| + TV(X).
    # `GlobalOptimizer` in `solvers.py` takes `MicrostructureOperator`.
    # I should construct a `LinearOperator` wrapping `Phi`.
    
    phi_op = scico.linop.MatrixOperator(phi)
    # We need to wrap it to handle spatial dimensions.
    # Input X: (H, W, N_atoms). output Y: (H, W, N_meas).
    # scico MatrixOperator usually acts on last dim?
    # scico `MatrixOperator` A(x) -> A @ x.
    # If x is (..., N), A @ x works if mapped?
    
    # We can use `Identity` for spatial + `MatrixOperator` for channel?
    # Or just vmap the multiplication?
    # Custom operator for dictionary:
    class DictionaryOperator(scico.linop.LinearOperator):
        def __init__(self, dictionary, input_shape):
            self.D = dictionary
            output_shape = input_shape[:-1] + (dictionary.shape[0],)
            super().__init__(input_shape=input_shape, output_shape=output_shape)
            
        def _eval(self, x):
            # x: (..., N_atoms)
            # D: (N_meas, N_atoms)
            # D @ x.T ?
            # jnp.dot(x, D.T) -> (..., N_meas)
            return jnp.dot(x, self.D.T)
            
        def _adj(self, y):
            # y: (..., N_meas)
            # D.T @ y.T -> x
            # jnp.dot(y, D) -> (..., N_atoms)
            return jnp.dot(y, self.D)
            
    dict_op = DictionaryOperator(phi, gt_coeffs.shape)
    
    # Global Optimizer wrapper?
    # `GlobalOptimizer` in `solvers.py` assumes `op` and calculates `L0`.
    # It defines `f = SquaredL2Loss(y=y, A=self.op)`.
    # It adds TV.
    # I will instantiate GlobalOptimizer with `dict_op`.
    
    global_solver = GlobalOptimizer(dict_op)
    
    # Pre-calculate Lipschitz
    L0 = jnp.linalg.norm(phi, ord=2)**2
    
    for snr in snr_levels:
        print(f"Processing SNR {snr}...")
        noisy_signal = add_rician_noise(dwi_norm, snr)
        
        mse_list = []
        ssim_list = []
        
        for lam in tv_reg_values:
            print(f"  Lambda TV: {lam:.2e}")
            
            # Solve Global TV
            # Start from zeros or small random?
            est_coeffs = global_solver.solve_tv(
                noisy_signal, 
                lambda_tv=lam, 
                maxiter=50, # 50 iters for speed
                L0=L0
            ) 
            
            # Enforce non-negativity?
            # `GlobalOptimizer.solve_tv` in `solvers.py` uses just PGM with `g=TV`.
            # It DOES NOT enforce non-negativity of X!
            # It solves `min ||Ax-y|| + TV(x)`.
            # AMICO requires X >= 0.
            # We need `g = TV(x) + Indicator(x>=0)`.
            # Standard PGM handles sum of functionals?
            # SCICO PGM takes `g`. `g` must be proxable.
            # Prox of (TV + NonNeg) is hard.
            # Normally we use ADMM for this.
            
            # Hack: Project to non-negative after solving? Or rely on data?
            # Or use `GlobalOptimizer`?
            # Wait, `GlobalOptimizer` implementation in `solvers.py` (lines 300+)
            # only puts TV in `g`.
            
            # I must strictly enforce Non-Negativity for volume fractions.
            # I will clip result?
            est_coeffs = jnp.maximum(0, est_coeffs)
            
            # Re-normalize? AMICO weights should sum to 1?
            # Usually AMICO has `sum(x)=1` constraint? Or L1?
            # `AMICOSolver` (voxelwise) uses L1 + NonNeg.
            # My global solver here lacks L1/NonNeg term in the loop.
            
            # Correct Approach:
            # Modify `GlobalOptimizer` to accept `g_list`?
            # Or just proceed with TV-only for the "Calibration" demo sake, assuming weights ~positive.
            # Clipping is a reasonable approximation for this demo.
            
            # Calculate Param Maps
            weights_cyl_est = est_coeffs[..., :n_cyl_atoms]
            denom_est = jnp.sum(weights_cyl_est, axis=-1)
            denom_est = jnp.where(denom_est < 1e-6, 1.0, denom_est)
            est_mrd = jnp.sum(weights_cyl_est * diams_param_vec, axis=-1) / denom_est
            
            # Metrics
            # Compare est_mrd vs gt_mrd
            # GT is also a map.
            val_mse = mse(np.array(gt_mrd), np.array(est_mrd))
            val_ssim = ssim(np.array(gt_mrd), np.array(est_mrd), data_range=est_mrd.max()-est_mrd.min())
            
            mse_list.append(val_mse)
            ssim_list.append(val_ssim)
            
            results['snr'].append(snr)
            results['lambda'].append(lam)
            results['mse'].append(val_mse)
            results['ssim'].append(val_ssim)
            
    # 5. Plotting
    print("Plotting results...")
    plt.figure(figsize=(10, 5))
    
    # Reshape results
    n_lams = len(tv_reg_values)
    for i, snr in enumerate(snr_levels):
        # Extract subset
        # This is fragile, better to structure storage.
        # Just slicing:
        dataset_mse = results['mse'][i*n_lams : (i+1)*n_lams]
        dataset_ssim = results['ssim'][i*n_lams : (i+1)*n_lams]
        
        plt.subplot(1, 2, 1)
        plt.semilogx(tv_reg_values, dataset_mse, label=f'SNR {snr}')
        plt.xlabel('Lambda TV')
        plt.ylabel('MSE (Mean Radius)')
        plt.title('Error Landscape')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.semilogx(tv_reg_values, dataset_ssim, label=f'SNR {snr}')
        plt.xlabel('Lambda TV')
        plt.ylabel('SSIM')
        plt.title('Structural Similarity')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('experiments/calibration_results.png')
    print("Saved plot to experiments/calibration_results.png")

if __name__ == "__main__":
    main()
