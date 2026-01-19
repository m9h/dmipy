
import numpy as np
import scipy.ndimage
import nibabel as nib

def generate_gaussian_random_field(shape, fwhm=5.0):
    """
    Generates a 3D Gaussian Random Field (GRF).
    
    Args:
        shape (tuple): (Nx, Ny, Nz)
        fwhm (float): Full Width at Half Maximum of the smoothing kernel (in voxels).
                      Controls the spatial correlation length.
    
    Returns:
        np.ndarray: Normalized GRF with mean 0 and std 1.
    """
    # 1. Generate white noise
    noise = np.random.normal(0, 1, shape)
    
    # 2. Smooth it
    # sigma = fwhm / (2 * sqrt(2 * ln(2))) ~= fwhm / 2.355
    sigma = fwhm / 2.355
    smooth_noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
    
    # 3. Normalize to standard normal
    smooth_noise -= np.mean(smooth_noise)
    smooth_noise /= np.std(smooth_noise)
    
    return smooth_noise

def generate_spatially_varying_snr_map(shape, base_snr=30, variation_magnitude=10):
    """
    Generates an SNR map that varies spatially using a GRF.
    This simulates coil sensitivity profiles or heterogeneous noise.
    """
    grf = generate_gaussian_random_field(shape, fwhm=10.0)
    
    # Scale GRF to variation range: e.g. -10 to +10
    snr_map = base_snr + grf * variation_magnitude
    
    # Clip to avoid negative SNR
    snr_map = np.maximum(snr_map, 5.0)
    
    return snr_map

def add_spatially_correlated_noise(signal, snr_map):
    """
    Adds Rician noise where the noise level (sigma) determines the local SNR.
    
    Args:
        signal (np.ndarray): (Nx, Ny, Nz, N_meas) specific signal.
        snr_map (np.ndarray): (Nx, Ny, Nz) or broadcastable.
    """
    # Sigma map = S0 / SNR. Assuming signal is normalized or we use max signal?
    # Usually SNR is defined relative to b0 intensities.
    # Let's assume signal approximately 1.0 at b0.
    
    sigma_map = 1.0 / snr_map
    # Broadcast to measurements
    sigma_map = sigma_map[..., None]
    
    noise_r = np.random.normal(0, 1, signal.shape) * sigma_map
    noise_i = np.random.normal(0, 1, signal.shape) * sigma_map
    
    noisy_signal = np.sqrt((signal + noise_r)**2 + noise_i**2)
    return noisy_signal

if __name__ == "__main__":
    dims = (20, 20, 20)
    print(f"Generating noise fields for {dims} volume...")
    
    # 1. Texture / Structure Noise (e.g. for parameter maps)
    # Simulate a "patchy" volume fraction map
    structure_grf = generate_gaussian_random_field(dims, fwhm=3.0)
    # Map to 0-1
    f_map = (structure_grf - structure_grf.min()) / (structure_grf.max() - structure_grf.min())
    print("Generated structural parameter map (range 0-1).")
    
    # 2. Coil Sensitivity / SNR Map
    snr_map = generate_spatially_varying_snr_map(dims, base_snr=20, variation_magnitude=5)
    print(f"Generated SNR map (mean {np.mean(snr_map):.2f}).")
    
    # 3. Fake Signal
    signal = np.ones(dims + (10,))
    
    # 4. Add Noise
    noisy = add_spatially_correlated_noise(signal, snr_map)
    
    print("Saving example noise fields...")
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(f_map, affine), 'example_structure_grf.nii.gz')
    nib.save(nib.Nifti1Image(snr_map, affine), 'example_snr_map.nii.gz')
    nib.save(nib.Nifti1Image(noisy, affine), 'example_noisy_signal.nii.gz')
    
    print("Done.")
