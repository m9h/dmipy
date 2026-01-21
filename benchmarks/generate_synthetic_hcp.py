
"""
Generator for the 'Synesthesia' Dataset (Synthetic Connectome 2.0).

Mimics the acquisition protocol of Connectome 2.0 (Massive b-values up to 17,800)
to test 'dmipy-jax' features in extreme conditions.
"""

import numpy as np
import jax.numpy as jnp
from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models import cylinder_models, gaussian_models, zeppelin

def generate_synthetic_connectome(shape=(20, 20, 5), snr=30):
    print("Generating Synthetic Connectome 2.0 Data...")
    
    # 1. Acquisition Definition (Connectome 2.0 approximation)
    # Protocol: 
    #   Short D (~13ms): b = 1000, 3000, 6000
    #   Long D (~30ms): b = 2000, 6000, 10000, 17800
    
    shells_short = [1000e6, 3000e6, 6000e6]
    shells_long  = [2000e6, 6000e6, 10000e6, 17800e6]
    
    N_dirs = 32 # Per shell for speed (Real C2.0 has ~64-100)
    
    bvals = []
    bvecs = []
    deltas = []
    Deltas = []
    
    # Generate directions (random on sphere)
    np.random.seed(42)
    def get_dirs(N):
        vecs = np.random.randn(N, 3)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    # Short D block
    for b in shells_short:
        bvals.append(np.full(N_dirs, b))
        bvecs.append(get_dirs(N_dirs))
        deltas.append(np.full(N_dirs, 0.007)) # 7ms
        Deltas.append(np.full(N_dirs, 0.013)) # 13ms
        
    # Long D block
    for b in shells_long:
        bvals.append(np.full(N_dirs, b))
        bvecs.append(get_dirs(N_dirs))
        deltas.append(np.full(N_dirs, 0.010)) # 10ms
        Deltas.append(np.full(N_dirs, 0.030)) # 30ms

    # Combine
    bvals = np.concatenate(bvals)
    bvecs = np.concatenate(bvecs)
    deltas = np.concatenate(deltas)
    Deltas = np.concatenate(Deltas)
    
    scheme = JaxAcquisition(bvals, bvecs, delta=deltas, Delta=Deltas)
    print(f"Acquisition: {len(bvals)} measures. Max b={bvals.max()/1e9:.1f}k")
    
    # 2. Phantom Anatomy
    # 20x20x5 slice
    # Zone 1 (Left): Single Fiber, Varying Diameter (2um -> 8um)
    # Zone 2 (Center): Crossing Fibers (90 deg)
    # Zone 3 (Right): CSF / Isotropic
    
    N_voxels = np.prod(shape)
    
    # Parameter Maps
    # Use standard strings for params
    mu_map = np.zeros(shape + (2,)) # theta, phi
    diam_map = np.zeros(shape + (1,))
    f_intra_map = np.zeros(shape + (1,))
    
    # Coordinates
    X, Y, Z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    
    # Defaults
    diam_map[:] = 3e-6
    f_intra_map[:] = 0.6
    mu_map[..., 0] = np.pi/2 # Theta=90 (in XY plane)
    mu_map[..., 1] = 0.0     # Phi=0 (Along X)
    
    # Zone 1: Diameter Gradient along Y (Left side X < 7)
    mask_z1 = (X < 7)
    # Map Y (0..19) to Diameter (2e-6 .. 8e-6)
    diam_map[mask_z1, 0] = 2e-6 + (Y[mask_z1] / 19.0) * 6e-6
    
    # Zone 2: Crossing (Center 7 <= X < 14)
    mask_z2 = (X >= 7) & (X < 14)
    # We need a crossing model here.
    # For sim, we can't easily switch model structure per voxel in 'simulate_signal' 
    # unless we use a Universal Model (2 sticks always).
    # Let's assume the whole image is modeled as 2-sticks, 
    # but in Zone 1/3 the second stick has f=0.
    
    # Setup Universal Ground Truth Model
    # 1. RestrictedCylinder (Main Fiber)
    # 2. RestrictedCylinder (Crossing Fiber) - Fixed diameter?
    # 3. Ball (CSF)
    # 4. Zeppelin (Extra-axonal) - coupled to Fiber 1? 
    # To keep it manageable:
    # Model = RestrCyl_1 + RestrCyl_2 + Ball
    
    cyl1 = cylinder_models.RestrictedCylinder()
    cyl2 = cylinder_models.RestrictedCylinder()
    ball = gaussian_models.Ball()
    mt_model = JaxMultiCompartmentModel([cyl1, cyl2, ball])
    
    # Build Parameter Dict
    params = {}
    
    # Fiber 1 (Main, Horizontal) -> 'mu', 'diameter', 'diffusivity_parallel' (Idx 0)
    params['mu'] = np.zeros(shape + (2,))
    params['mu'][..., 0] = np.pi/2
    params['mu'][..., 1] = 0.0
    params['diameter'] = diam_map
    params['lambda_par'] = np.full(shape + (1,), 1.7e-9)
    
    # Fiber 2 (Crossing, Vertical) -> 'mu_2', 'diameter_2', 'diffusivity_parallel_2' (Idx 1)
    params['mu_2'] = np.zeros(shape + (2,))
    params['mu_2'][..., 0] = np.pi/2
    params['mu_2'][..., 1] = np.pi/2
    params['diameter_2'] = np.full(shape + (1,), 3e-6) # Fixed 3um crossing
    params['lambda_par_2'] = np.full(shape + (1,), 1.7e-9)
    
    # Fractions
    # Initialize
    f_1 = np.zeros(shape)
    f_2 = np.zeros(shape)
    f_csf = np.zeros(shape)
    
    # Zone 1 (Left): Single Fiber
    # f1 = 0.7, f2 = 0, f_csf = 0.3
    f_1[mask_z1] = 0.7
    f_2[mask_z1] = 0.0
    f_csf[mask_z1] = 0.3
    
    # Zone 2 (Center): Crossing
    # f1 = 0.35, f2 = 0.35, f_csf = 0.3
    f_2[mask_z2] = 0.35
    f_1[mask_z2] = 0.35
    f_csf[mask_z2] = 0.3
    
    # Zone 3 (Right): CSF Dominant
    mask_z3 = (X >= 14)
    f_1[mask_z3] = 0.1
    f_2[mask_z3] = 0.0
    f_csf[mask_z3] = 0.9
    
    params['partial_volume_0'] = f_1[..., None]
    params['partial_volume_1'] = f_2[..., None]
    params['partial_volume_2'] = f_csf[..., None]
    
    params['lambda_iso'] = np.full(shape + (1,), 3e-9)
    
    print("Simulating Signal...")
    # Using JAX to simulate
    # Flatten spatial dims for MCM compatibility (expects N_vox, params)
    N_vox = np.prod(shape)
    params_flat = {}
    for k, v in params.items():
        # v is (X, Y, Z, ...)
        # Flatten first 3 dims
        v_flat = v.reshape((N_vox,) + v.shape[3:]) 
        # Check if last dim is 1 and squeeze? MCM handles (1,) usually?
        # parameter_dictionary_to_array handles (1,) for scalar.
        params_flat[k] = v_flat

    # Call model
    # Output: (N_vox, N_meas)
    signal_flat = mt_model(params_flat, scheme)
    
    # Reshape back: (X, Y, Z, N_meas)
    signal = signal_flat.reshape(shape + (len(bvals),))
    
    # Add Noise
    noise_sigma = 1.0 / snr
    noise_real = np.random.normal(0, noise_sigma, signal.shape)
    noise_imag = np.random.normal(0, noise_sigma, signal.shape)
    
    signal_noisy = np.sqrt((signal + noise_real)**2 + noise_imag**2)
    
    print("Data Generation Complete.")
    return signal_noisy, scheme, params

if __name__ == "__main__":
    data, scheme, gt = generate_synthetic_connectome()
    # Save to disk if needed, or used by import
