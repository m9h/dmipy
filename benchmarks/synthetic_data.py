
import numpy as np
import nibabel as nib
import os
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.signal_models.cylinder_models import C2CylinderStejskalTannerApproximation
from dmipy.signal_models.gaussian_models import G1Ball

def generate_hcp_b_table(shells, directions_per_shell, b0_count=20):
    """
    Generates a synthetic HCP-like b-table.
    
    Args:
        shells (list): List of b-values (s/mm^2).
        directions_per_shell (list): List of number of directions per shell.
        b0_count (int): Number of b0 images.
        
    Returns:
        bvals, bvecs (numpy.ndarray)
    """
    bvals = []
    bvecs = []
    
    # Add b0s
    for _ in range(b0_count):
        bvals.append(0.0)
        bvecs.append([0.0, 0.0, 0.0])
        
    for shell_b, n_dirs in zip(shells, directions_per_shell):
        # Generate uniform directions on sphere
        # Using a simple deterministic spiral or similar if dmipy util unavailable
        # But dmipy has spherical_design? Let's try to use dipy or just a helper.
        # We'll use a golden section spiral for uniform distribution
        
        indices = np.arange(0, n_dirs, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_dirs)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        vecs = np.stack([x, y, z], axis=1)
        
        for v in vecs:
            bvals.append(shell_b)
            # Normalize just in case
            v = v / np.linalg.norm(v)
            bvecs.append(v)
            
    return np.array(bvals), np.array(bvecs)

def create_phantom_volume(dim=100):
    """
    Creates a 100x100x100 phantom (default) with Core, Transition, and Shell regions.
    
    Returns:
        volume geometry and parameter maps
    """
    x = np.linspace(-1, 1, dim)
    y = np.linspace(-1, 1, dim)
    z = np.linspace(-1, 1, dim)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    R = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Initialize Masks
    # Core: R < 0.3
    # Transition: 0.3 <= R < 0.8
    # Shell: R >= 0.8
    # Typically we might want R to go up to 1. 0.8 to 1.0 is shell.
    # But box is -1 to 1. Corners are R > 1.
    
    core_mask = R < 0.3
    trans_mask = (R >= 0.3) & (R < 0.8)
    shell_mask = R >= 0.8
    
    # Initialize Parameter Maps
    dims = (dim, dim, dim)
    
    # Fractions
    f_cyl1 = np.zeros(dims)
    f_cyl2 = np.zeros(dims)
    f_iso = np.zeros(dims)
    
    # Microstructure Params
    diameter_map = np.zeros(dims)
    
    # Core: Crossing (mu=[1,0,0] and [0,1,0]) @ 90 deg
    # Assume equal fractions 0.5, 0.5
    f_cyl1[core_mask] = 0.5
    f_cyl2[core_mask] = 0.5
    # Diameter in core: Let's assume constant equivalent to transition start?
    # Transition is 2um to 8um.
    # Let's say Core is 4um or 2um.
    # "Varying axon diameters from 2um to 8um" usually implies growth outward.
    # So start at 2um at R=0.3 -> 8um at R=0.8?
    # R_norm = (R - 0.3) / (0.8 - 0.3) = (R-0.3)/0.5
    # d = 2 + 6 * R_norm
    
    # Core diameter: 2um
    diameter_map[core_mask] = 2.0 * 1e-6
    
    # Transition
    # Assume crossing persists? "Varying axon diameters" refers to fibers.
    # If we want a nice transition, we can keep crossing.
    f_cyl1[trans_mask] = 0.5
    f_cyl2[trans_mask] = 0.5
    
    r_vals = R[trans_mask]
    d_vals = 2.0 + (r_vals - 0.3) / 0.5 * 6.0 # 2 to 8
    diameter_map[trans_mask] = d_vals * 1e-6
    
    # Shell: Isotropic
    f_iso[shell_mask] = 1.0
    f_cyl1[shell_mask] = 0.0
    f_cyl2[shell_mask] = 0.0
    # Diameter doesn't matter in shell (f=0)
    
    return {
        'f_cyl1': f_cyl1,
        'f_cyl2': f_cyl2,
        'f_iso': f_iso,
        'diameter': diameter_map,
        'R': R
    }

def simulate_signal(scheme, params, batch_size=10000):
    """
    Simulates signal for the given scheme and parameter maps using dmipy models.
    Uses batching to avoid OOM.
    """
    bvals = scheme.bvalues
    bvecs = scheme.gradient_directions
    # Flatten parameter maps
    f_cyl1_ = params['f_cyl1'].ravel()
    f_cyl2_ = params['f_cyl2'].ravel()
    f_iso_ = params['f_iso'].ravel()
    diameter_ = params['diameter'].ravel()
    
    n_vox = f_cyl1_.size
    n_meas = len(bvals)
    
    S_total = np.zeros((n_vox, n_meas), dtype=np.float32)
    
    # Constants
    lambda_par = 1.7e-9
    lambda_iso = 3.0e-9
    
    # Scheme arrays for broadcasting
    bvals_ = bvals[None, :] # (1, N_meas)
    q_ = scheme.qvalues[None, :]
    n_ = bvecs 
    
    # Models
    mu1 = np.array([1., 0., 0.]) # Cartesian X
    mu2 = np.array([0., 1., 0.]) # Cartesian Y
    # Note: mu1/mu2 in Helper expects Cartesian now? 
    # _compute_cylinder_signal expects Cartesian mu.
    
    # Helper instantiations (dummy)
    # We pass the class or just use the helper logic directly.
    # The helper logic uses scipy.special.j1.
    
    print(f"Simulating {n_vox} voxels in batches of {batch_size}...")
    
    for start_idx in range(0, n_vox, batch_size):
        end_idx = min(start_idx + batch_size, n_vox)
        
        # Batch parameters
        d_batch = diameter_[start_idx:end_idx][:, None] # (B, 1)
        f1_batch = f_cyl1_[start_idx:end_idx][:, None]
        f2_batch = f_cyl2_[start_idx:end_idx][:, None]
        f_iso_batch = f_iso_[start_idx:end_idx][:, None]
        
        # Compute Signals
        E_cyl1 = _compute_cylinder_signal_batch(bvals_, q_, n_, mu1, lambda_par, d_batch)
        E_cyl2 = _compute_cylinder_signal_batch(bvals_, q_, n_, mu2, lambda_par, d_batch)
        E_ball = np.exp(-bvals_ * lambda_iso) # (1, N_meas) broadcasted automatically
        
        # Combine
        S_batch = f1_batch * E_cyl1 + f2_batch * E_cyl2 + f_iso_batch * E_ball
        
        S_total[start_idx:end_idx, :] = S_batch
        
        if (start_idx // batch_size) % 5 == 0:
            print(f"  Processed {end_idx}/{n_vox} voxels...")
            
    dims = params['f_cyl1'].shape
    return S_total.reshape(dims + (n_meas,))

def _compute_cylinder_signal_batch(bvals, q, n, mu, lambda_par, diameter):
    # diameter: (B, 1)
    # q: (1, N_meas)
    # n: (N_meas, 3)
    # mu: (3,)
    
    # Parallel
    dot_prod = np.dot(n, mu) # (N_meas,)
    E_parallel = np.exp(-bvals * lambda_par * dot_prod**2) # (1, N_meas)
    
    # Perpendicular
    mu_outer = np.outer(mu, mu)
    idx = np.eye(3) - mu_outer
    mag_perp = np.linalg.norm(np.dot(idx, n.T), axis=0) # (N_meas,)
    q_perp = q * mag_perp # (1, N_meas)
    
    radius = diameter / 2.0 # (B, 1)
    arg = 2 * np.pi * q_perp * radius # (B, N_meas)
    
    safe_arg = np.where(arg == 0, 1.0, arg)
    
    from scipy.special import j1
    E_perp = (2 * j1(safe_arg) / safe_arg) ** 2
    E_perp[arg == 0] = 1.0
    
    return E_parallel * E_perp

def add_noise_batched(signal, snr=30, batch_size=50000):
    """
    Adds Rician noise to signal in place (or minimal copy) to save memory.
    """
    sigma = 1.0 / snr
    n_vox = signal.shape[0]
    n_meas = signal.shape[1]
    
    # Process in batches
    # We can modify signal in-place if allowed.
    # But usually we return new array. To save memory, we can create output array 
    # and fill it, or modify input if we don't need clean signal anymore.
    # Let's modify in-place to be most efficient if 'signal' is not needed later.
    
    print(f"Adding noise to {n_vox} voxels in batches...")
    
    for start in range(0, n_vox, batch_size):
        end = min(start + batch_size, n_vox)
        
        # Sub-slice
        S_chunk = signal[start:end]
        
        # Generate noise
        n1 = np.random.normal(0, sigma, S_chunk.shape)
        n2 = np.random.normal(0, sigma, S_chunk.shape)
        
        # Rice
        # S_noisy = sqrt((S+n1)^2 + n2^2)
        # In-place update: signal[start:end] = ...
        
        # Temp calc
        noisy_chunk = np.sqrt((S_chunk + n1)**2 + n2**2)
        
        # Write back
        signal[start:end] = noisy_chunk
        
        if (start // batch_size) % 5 == 0:
             print(f"  Noise added to {end}/{n_vox} voxels...")
             
    return signal

def main():
    print("Generating Synthetic HCP Benchmark Data...")
    
    # 1. B-Table
    print("1. Generating B-Table (HCP-MGH Replicate)...")
    shells = [1000, 3000, 5000, 10000]
    dirs = [64, 64, 128, 256]
    bvals, bvecs = generate_hcp_b_table(shells, dirs)
    
    # Acq Params
    Delta = 21.8e-3
    delta = 12.9e-3
    
    scheme = acquisition_scheme_from_bvalues(
        bvals * 1e6, bvecs, delta, Delta
    )
    
    # 2. Phantom
    dim = 100
    print(f"2. Creating Phantom Geometry ({dim}x{dim}x{dim})...")
    geom_params = create_phantom_volume(dim=dim)
    
    # 3. Simulate
    print("3. Simulating Signal (Noise-free) with batching...")
    # S_clean is now a large array (1M, 532).
    # We flattened it in simulate_signal return.
    # Wait, simulate_signal returns reshaped (dim, dim, dim, n_meas)
    # add_noise_batched expects (n_vox, n_meas)?
    # We should probably work with flattened array for noise, then reshape.
    
    # Let's modify simulate_signal to return flattened, or flatten here.
    # Actually simulate_signal returns (100,100,100,532).
    # We can flatten view.
    
    S_clean = simulate_signal(scheme, geom_params, batch_size=10000)
    
    # Flatten view for noise addition
    # S_flat = S_clean.reshape(-1, S_clean.shape[-1])
    # But reshape might copy if not contiguous. 
    # Usually it is.
    
    # 4. Add Noise
    print(f"4. Adding Rician Noise (SNR=30)...")
    # We pass the flattened view. It modifies data in place (if view is valid).
    # To ensures it modifies the original S_clean, we should verify.
    # Or just re-assign.
    
    # To be safe, let's keep it flattened for processing and reshape for save.
    
    # Actually, let's handle the shape inside add_noise_batched or just flatten here.
    n_meas = S_clean.shape[-1]
    S_flat = S_clean.reshape(-1, n_meas)
    
    add_noise_batched(S_flat, snr=30, batch_size=50000)
    
    # S_clean now contains noisy data (in place modification via S_flat view).
    
    # 5. Save
    print("5. Saving Files to current directory...")
    np.savetxt('synthetic.bval', bvals[None, :], fmt='%d')
    np.savetxt('synthetic.bvec', bvecs.T, fmt='%.6f')
    
    affine = np.eye(4)
    # S_clean is already 4D (100,100,100,532)
    img = nib.Nifti1Image(S_clean.astype(np.float32), affine)
    nib.save(img, 'synthetic_hcp.nii.gz')
    
    print("Done! Output: synthetic_hcp.nii.gz, synthetic.bval, synthetic.bvec")

if __name__ == "__main__":
    main()
