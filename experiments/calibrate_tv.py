
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from dmipy_jax.io.connectome2 import load_connectome2_mri
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.core.acquisition import SimpleAcquisitionScheme as JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import RestrictedCylinder
from dmipy_jax.signal_models.sphere_models import SphereGPD
from dmipy_jax.inverse.global_amico import GlobalAMICOSolver

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
    dwi = jnp.squeeze(dwi, axis=0) # (40, 40, N_meas)
    print(f"Data Shape: {dwi.shape}")

    # Normalize by approx b0
    b0_mask = scheme.bvalues < 50
    b0_map = jnp.mean(dwi[..., b0_mask], axis=-1, keepdims=True)
    dwi_norm = dwi / jnp.maximum(b0_map, 1e-6)

    # 2. Setup AMICO Model
    print("Initializing AMICO Dictionary...")
    
    # Re-wrap scheme into JaxAcquisition with timing
    # Typically Connectome 2.0: Delta=43ms, delta=12ms approx ? 
    # Or just use arbitrary values valid for restricted/sphere models.
    # We'll assume typical clinical/research values.
    acq = JaxAcquisition(
        scheme.bvalues, 
        scheme.gradient_directions,
        delta=0.012, 
        Delta=0.043,
        b0_threshold=50
    )

    # Directions for restricted cylinder
    n_dirs = 32
    dirs_cart = get_fibonacci_sphere(n_dirs)
    dirs_sphere = cart2sphere(dirs_cart)
    mu_grid = [d for d in dirs_sphere] # List of (2,) arrays (theta, phi)

    # Diameters
    diameters = np.linspace(1e-6, 8e-6, 6) # 1 to 8 microns

    mc_model = JaxMultiCompartmentModel([
        RestrictedCylinder(lambda_par=1.7e-9), 
        SphereGPD(diffusion_constant=3.0e-9, diameter=15e-6) # Large sphere as free water proxy
    ])
    
    # Handle parameter naming for grid
    rc_diam_name = 'diameter_1' if 'diameter_1' in mc_model.parameter_names else 'diameter'
        
    grid = {
        'mu': mu_grid,
        rc_diam_name: diameters
    }
    
    # Initialize Global Solver
    solver = GlobalAMICOSolver(mc_model, acq, grid)
    print(f"Dictionary Size: {solver.dict_matrix.shape}")
    
    # 3. Establish Ground Truth (High SNR Fit)
    print("Fitting Ground Truth (High SNR)...")
    
    # Use global solver with small L1 for ground truth, TV=0
    # Or just use fit_global with TV=0
    gt_coeffs = solver.fit_global(
        dwi_norm, 
        lambda_tv=0.0, 
        lambda_l1=1e-3, 
        maxiter=100
    )
    # gt_coeffs: (40, 40, N_atoms) assumed since dwi_norm is (40,40, N_meas)
    
    # Calculate Mean Diameter Map for Ground Truth
    # Need to reconstruct diameter vector per atom
    # Order in generate_kernels: Loop submodels -> Loop combinations
    # Model 1 (RC): itertools.product(mu, diameter)
    # Model 2 (Sphere): single atom
    
    n_mu = len(mu_grid)
    n_diam = len(diameters)
    n_cyl_atoms = n_mu * n_diam
    
    # Create vector of diameters for atoms
    # itertools order: mu varies for EACH diameter? Or diameter varies for each mu?
    # global_amico.py: `combinations = list(itertools.product(*val_lists))`
    # val_lists order depends on `sub_model_params` order.
    # RestrictedCylinder params: mu, lambda_par, diameter
    # If using default order: mu, lambda_par, diameter.
    # But dictionary_params has 'mu', 'diameter'. 'lambda_par' is fixed.
    # param_order depends on sub_model loop.
    # Usually: 'mu', 'lambda_par', 'diameter'.
    # So product(mu_grid, [lam], diameters)
    # mu is first -> Slowest varying axis.
    # diameter is last -> Fastest varying axis.
    # So for each mu, we cycle through all diameters.
    
    # diams_per_atom pattern: 
    # [d1, d2, d3, d4, d5, d6, d1, d2...]
    diams_per_atom = np.tile(diameters, n_mu)
    diams_param_vec = jnp.array(diams_per_atom)
    
    def calc_mean_diameter(coeffs):
        # Extract cylinder coeffs
        # coeffs: (..., N_total)
        w_cyl = coeffs[..., :n_cyl_atoms]
        
        denom = jnp.sum(w_cyl, axis=-1)
        denom = jnp.where(denom < 1e-6, 1.0, denom)
        
        mrd = jnp.sum(w_cyl * diams_param_vec, axis=-1) / denom
        return mrd

    gt_mrd = calc_mean_diameter(gt_coeffs)
    
    # 4. Calibration Loop
    snr_levels = [30] # Just one SNR for speed
    tv_reg_values = [0.001, 0.01, 0.1, 0.5] # Fewer points
    
    results = {
        'snr': [],
        'lambda': [],
        'mse': [],
        'ssim': []
    }
    
    # Pre-calculate L0 (Lipschitz) once? 
    # GlobalAMICOSolver does it internally per fit call.
    
    gt_mrd_np = np.array(gt_mrd)
    data_range = 10e-6 # Fixed range (0 to 10 microns)
    
    for snr in snr_levels:
        print(f"Processing SNR {snr}...")
        noisy_signal = add_rician_noise(dwi_norm, snr)
        
        for lam in tv_reg_values:
            print(f"  Lambda TV: {lam:.2e}")
            
            est_coeffs = solver.fit_global(
                noisy_signal, 
                lambda_tv=lam, 
                lambda_l1=1e-3,
                maxiter=30, # Decrease iter for speed
                display=False
            )
            
            est_mrd = calc_mean_diameter(est_coeffs)
            est_mrd_np = np.array(est_mrd)
            
            # Metrics
            # Compare est_mrd vs gt_mrd
            val_mse = mse(gt_mrd_np, est_mrd_np)
            
            # Use fixed data_range for SSIM to avoid drift/instability
            val_ssim = ssim(gt_mrd_np, est_mrd_np, data_range=data_range)
            
            print(f"    MSE: {val_mse:.2e}, SSIM: {val_ssim:.4f}")
            
            results['snr'].append(snr)
            results['lambda'].append(lam)
            results['mse'].append(val_mse)
            results['ssim'].append(val_ssim)
            
    # 5. Plotting
    print("Plotting results...")
    plt.figure(figsize=(12, 5))
    
    n_lams = len(tv_reg_values)
    
    for i, snr in enumerate(snr_levels):
        start_idx = i * n_lams
        end_idx = (i + 1) * n_lams
        
        curr_lams = results['lambda'][start_idx:end_idx]
        curr_mse = results['mse'][start_idx:end_idx]
        curr_ssim = results['ssim'][start_idx:end_idx]
        
        plt.subplot(1, 2, 1)
        plt.semilogx(curr_lams, curr_mse, 'o-', label=f'SNR {snr}')
        
        plt.subplot(1, 2, 2)
        plt.semilogx(curr_lams, curr_ssim, 'o-', label=f'SNR {snr}')
        
    plt.subplot(1, 2, 1)
    plt.xlabel('Lambda TV')
    plt.ylabel('MSE (Mean Diameter)')
    plt.title('Error Landscape')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Lambda TV')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('experiments/calibration_results.png')
    print("Saved plot to experiments/calibration_results.png")

if __name__ == "__main__":
    main()
