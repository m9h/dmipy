
import numpy as np
import dmipy
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
import scipy.ndimage
import os

def get_acquisition_scheme(bval, n_dirs):
    """
    Generate an acquisition scheme with `n_dirs` encoded directions at `bval`.
    Also includes one b=0 image.
    """
    # Generate uniform directions using Fibonacci sphere
    def fibonacci_sphere(samples):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append((x, y, z))
        return np.array(points)

    vecs = fibonacci_sphere(n_dirs)
    bvals = np.ones(n_dirs) * bval
    bvecs = vecs
    
    # Add b0
    bvals = np.r_[0, bvals]
    bvecs = np.r_[np.array([[0, 0, 0]]), bvecs]
    
    # Create dmipy scheme
    scheme = acquisition_scheme_from_bvalues(bvals, bvecs, delta=0.01, Delta=0.03) # Standard pulse timings
    return scheme, bvals, bvecs

def generate_phantom_params(shape):
    """
    Generate parameter maps for a crossing fiber phantom.
    Shape: (Nx, Ny, Nz)
    
    Returns:
        gt_params: Dictionary of parameter maps
    """
    Nx, Ny, Nz = shape
    
    # Initialize parameters
    mu1 = np.zeros(shape + (3,))
    mu2 = np.zeros(shape + (3,))
    f1 = np.zeros(shape)
    f2 = np.zeros(shape)
    f_iso = np.zeros(shape)
    
    # Define Regions
    # Fiber 1: Horizontal (X-axis)
    mu1[..., 0] = 1.0 
    
    # Fiber 2: Vertical (Y-axis)
    mu2[..., 1] = 1.0
    
    # Setup volume fractions
    # Background (Isotropic only)
    f_iso[:] = 1.0
    
    # Crossing Region logic (Center)
    center_x, center_y = Nx // 2, Ny // 2
    width = Nx // 3
    
    x_coords = np.arange(Nx)
    y_coords = np.arange(Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Band 1 (Horizontal)
    mask1 = (np.abs(Y - center_y) < width // 2)
    # Band 2 (Vertical)
    mask2 = (np.abs(X - center_x) < width // 2)
    
    # Crossing
    crossing_mask = mask1 & mask2
    single_1_mask = mask1 & (~mask2)
    single_2_mask = mask2 & (~mask1)
    
    # Assign fractions
    # Crossing: 0.3 F1, 0.3 F2, 0.4 Iso
    f1[crossing_mask] = 0.3
    f2[crossing_mask] = 0.3
    f_iso[crossing_mask] = 0.4
    
    # Single 1: 0.6 F1, 0.4 Iso
    f1[single_1_mask] = 0.6
    f2[single_1_mask] = 0.0
    f_iso[single_1_mask] = 0.4
    
    # Single 2: 0.6 F2, 0.4 Iso
    f1[single_2_mask] = 0.0
    f2[single_2_mask] = 0.6
    f_iso[single_2_mask] = 0.4

    # Normalize roughly (already normalized above, but good practice)
    total = f1 + f2 + f_iso
    f1 /= total
    f2 /= total
    f_iso /= total
    
    params = {
        'mu1': mu1,
        'mu2': mu2,
        'f1': f1,
        'f2': f2,
        'f_iso': f_iso
    }
    return params

def main():
    print("Generating Synthetic Oracle Data...")
    
    # 1. Define HR Grid
    # 1mm isotropic. 50x50x10 FOV -> 5x5x1 cm.
    hr_shape = (50, 50, 5) 
    print(f"High Resolution Shape: {hr_shape}")
    
    # 2. Acquisition Protocols
    # HR: 90 dirs, b=3000
    bval = 3000e6 # SI units? dmipy uses SI usually. Let's stick to s/mm^2 for now and convert.
    # dmipy acquisition_scheme_from_bvalues takes bvals in s/mm^2 usually
    # Looking at dmipy docs/examples, standard is s/mm^2.
    
    bval_hr = 3000.0
    n_dirs_hr = 90
    scheme_hr, bvals_hr, bvecs_hr = get_acquisition_scheme(bval_hr, n_dirs_hr)
    
    # LR: 30 dirs, b=3000 (subset)
    # Actually, we should simulate at HR (90 dirs) then subsample dirs later? 
    # Or simulate LR acquisition directly? 
    # The requirement says "Downsamples this to Low Resolution... (LR images)".
    # Usually this means Spatial Downsampling.
    # But it also says "30 directions". 
    # So we will simulate 90 dirs HR, then spatially downsample, then subsample directions to 30.
    
    # 3. Define Model
    # C1Stick (x2) + Ball
    # Note: In dmipy, we can build a MultiCompartmentModel
    stick1 = cylinder_models.C1Stick()
    stick2 = cylinder_models.C1Stick()
    ball = gaussian_models.G1Ball()
    
    # Construct MC Model
    # We can't easily instantiate "Two Sticks" in dmipy identically without specific handling in the MC model construction
    # or passing different parameters.
    # Actually, dmipy models are defined by the compartments.
    # If we want two sticks, we can just use one Stick model but simulate it by summing signals manually 
    # OR use a specific MultiCompartmentModel structure.
    # Simplest way for generation: Simulate Signal 1 + Signal 2 + Signal 3.
    
    # Remove empty calls
    # S_stick = stick1(scheme_hr) 
    # S_ball = ball(scheme_hr)
    
    # helper to convert cartesian (..., 3) to (2, ...) theta, phi
    def vec2ang(mu):
        # mu: (..., 3)
        x, y, z = mu[..., 0], mu[..., 1], mu[..., 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero
        r[r==0] = 1.0
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        # stack to (2, ...)
        return np.stack([theta, phi], axis=0)

    # 4. Generate Parameters
    params = generate_phantom_params(hr_shape)
    
    # 5. Simulate HR Signals Manually (Robust to broadcasting)
    print("Simulating High Resolution Signals manually...")
    
    # D values
    D_par = 1.7e-3 # mm^2/s
    D_iso = 1.7e-3 # mm^2/s
    
    # Extract Scheme
    bvals = bvals_hr
    bvecs = bvecs_hr # (N_meas, 3)
    
    # Calculate Dot Products
    # mu: (Nx, Ny, Nz, 3)
    # bvecs: (N_meas, 3)
    # Result: (Nx, Ny, Nz, N_meas)
    dot1 = np.einsum('xyzi, mi -> xyzm', params['mu1'], bvecs)
    dot2 = np.einsum('xyzi, mi -> xyzm', params['mu2'], bvecs)
    
    # Stick Signals
    s1 = np.exp(-bvals * D_par * dot1**2)
    s2 = np.exp(-bvals * D_par * dot2**2)
    
    # Ball Signal
    # Isotropic: exp(-b * D)
    s_iso = np.exp(-bvals * D_iso) # (N_meas,)
    s_iso = np.tile(s_iso, hr_shape + (1,)) # Broadcast
    
    # Combine
    hr_signal = (params['f1'][..., None] * s1 + 
                 params['f2'][..., None] * s2 + 
                 params['f_iso'][..., None] * s_iso)
                 
    print(f"HR Signal Shape: {hr_signal.shape}")
    
    # 6. Downsample to Low Resolution
    # 1mm -> 2.5mm. Zoom factor = 1/2.5 = 0.4
    zoom_factor = 1.0 / 2.5
    # We downsample Spatial dims (0, 1, 2) but keep Measurements dim (3)
    print(f"Downsampling to Low Resolution (Zoom: {zoom_factor})...")
    
    # scipy.ndimage.zoom
    # Warning: zoom on large array can be slow. iterating over meas dimension might be safer memory-wise if huge, 
    # but 50x50x5x91 is tiny.
    lr_signal_full_dirs = scipy.ndimage.zoom(hr_signal, (zoom_factor, zoom_factor, zoom_factor, 1), order=1)
    
    print(f"LR Signal (Full Dirs) Shape: {lr_signal_full_dirs.shape}")
    
    # 7. Subsample Directions (90 -> 30)
    # We also keep b0 (index 0).
    # Simple strategy: keep index 0, then every 3rd direction of the 90.
    # indices: 0, 1, 4, 7...
    # 90 dirs + 1 b0 = 91 volumes.
    # indices 1..90 are DWI.
    # select 30 from them.
    dw_indices = np.linspace(1, 90, 30, dtype=int)
    keep_indices = np.concatenate(([0], dw_indices))
    
    lr_signal = lr_signal_full_dirs[..., keep_indices]
    lr_bvals = bvals_hr[keep_indices]
    lr_bvecs = bvecs_hr[keep_indices]
    
    print(f"LR Signal (Subsampled) Shape: {lr_signal.shape}")
    
    # 8. Save Data
    out_file = 'synthetic_oracle_data.npz'
    print(f"Saving to {out_file}...")
    np.savez_compressed(out_file, 
                        # HR Ground Truth
                        hr_signal=hr_signal,
                        hr_bvals=bvals_hr,
                        hr_bvecs=bvecs_hr,
                        gt_params=params, # dictionary inside npz
                        # LR Input
                        lr_signal=lr_signal,
                        lr_bvals=lr_bvals,
                        lr_bvecs=lr_bvecs,
                        # Metadata
                        hr_resolution=1.0,
                        lr_resolution=2.5)
                        
    print("Done.")

if __name__ == "__main__":
    main()
