import jax.numpy as jnp
from dmipy_jax.simulation.scanner.objects import IsochromatPhantom

def SyntheticBigMacPhantom(n_spins: int = 10000, radius: float = 0.05, seed: int = 42) -> IsochromatPhantom:
    """
    Generates a synthetic 'BigMac' phantom composed of multiple concentric tissue layers.
    
    This is a procedural generator since raw BigMac MRI data is not available locally.
    It mimics a brain-like structure with:
    1. Core: White Matter-like (Short T2, T1~800ms)
    2. Shell: Gray Matter-like (Longer T2, T1~1300ms)
    3. Outer: CSF-like (Long T1/T2)
    
    Args:
        n_spins: Total number of isochromats.
        radius: approximate radius of the phantom in meters.
        seed: Random seed for analytical distribution.
        
    Returns:
        IsochromatPhantom: The populated phantom object.
    """
    key = jnp.zeros(2, dtype=jnp.uint32) # Simple seed, or utilize proper PRNG key handling if JAX context is passed
    # For now, let's use numpy for generation to keep it simple, as this is effectively data loading
    import numpy as np
    np.random.seed(seed)
    
    # Generate random positions in a sphere
    # Rejection sampling for uniform sphere
    positions = []
    chunk = n_spins
    while len(positions) < n_spins:
        pts = np.random.uniform(-radius, radius, size=(chunk, 3))
        # Keep if inside radius
        dist_sq = np.sum(pts**2, axis=1)
        valid = pts[dist_sq <= radius**2]
        positions.extend(valid)
        
    positions = np.array(positions[:n_spins])
    radii = np.sqrt(np.sum(positions**2, axis=1))
    
    # Define layers
    # Core (WM): r < 0.5 * R
    # Shell (GM): 0.5 * R <= r < 0.9 * R
    # Outer (CSF): r >= 0.9 * R
    
    T1 = np.zeros(n_spins)
    T2 = np.zeros(n_spins)
    M0 = np.ones(n_spins)
    off_resonance = np.zeros(n_spins)
    
    r_norm = radii / radius
    
    # Masks
    mask_wm = r_norm < 0.5
    mask_gm = (r_norm >= 0.5) & (r_norm < 0.9)
    mask_csf = r_norm >= 0.9
    
    # Assign properties (approximate values at 3T)
    # WM: T1=800ms, T2=80ms
    T1[mask_wm] = 0.8
    T2[mask_wm] = 0.08
    M0[mask_wm] = 0.7 # WM has lower proton density
    
    # GM: T1=1300ms, T2=110ms
    T1[mask_gm] = 1.3
    T2[mask_gm] = 0.11
    M0[mask_gm] = 0.8
    
    # CSF: T1=4000ms, T2=2000ms
    T1[mask_csf] = 4.0
    T2[mask_csf] = 2.0
    M0[mask_csf] = 1.0
    
    # Add some random inhomogeneity to off-resonance
    # e.g. linear gradient + random
    off_resonance = np.random.normal(0, 10.0, size=n_spins) # Hz
    
    # Cast to JAX arrays
    return IsochromatPhantom(
        positions=jnp.array(positions, dtype=jnp.float32),
        T1=jnp.array(T1, dtype=jnp.float32),
        T2=jnp.array(T2, dtype=jnp.float32),
        M0=jnp.array(M0, dtype=jnp.float32),
        off_resonance=jnp.array(off_resonance, dtype=jnp.float32)
    )
    )

from dmipy_jax.io.datasets import load_bigmac_mri

def RealBigMacPhantom(dataset_path: str, downsample: int = 1) -> IsochromatPhantom:
    """
    Creates a phantom from the Real BigMac dataset.
    
    Args:
        dataset_path: Path to the BigMac data directory.
        downsample: Factor to downsample coordinates for lighter simulation (stride).
        
    Returns:
        IsochromatPhantom populated with real T1/T2/ProtonDensity.
    """
    data_dict = load_bigmac_mri(dataset_path)
    
    if 'T1' not in data_dict or 'T2' not in data_dict:
        raise ValueError("Dataset must contain T1 and T2 maps.")
        
    T1_map = data_dict['T1']
    T2_map = data_dict['T2']
    
    # Masking
    if 'mask' in data_dict:
        mask = data_dict['mask'] > 0
    else:
        # Generate simple mask from non-zero T1
        mask = T1_map > 0
        
    # Extract voxels
    # Create coordinate grid
    ndim = T1_map.ndim
    shape = T1_map.shape
    
    # Assuming standard orientation, or we need affine.
    # For simulation, we often center at 0.
    # Let's generate indices.
    x, y, z = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), jnp.arange(shape[2]), indexing='ij')
    
    # Flatten
    x_flat = x[mask]
    y_flat = y[mask]
    z_flat = z[mask]
    
    # Apply downsampling if needed to reduce spin count
    if downsample > 1:
        x_flat = x_flat[::downsample]
        y_flat = y_flat[::downsample]
        z_flat = z_flat[::downsample]
        
        # Need to re-mask properties
        # Actually easier to downsample indices
        
    positions = jnp.stack([x_flat, y_flat, z_flat], axis=1)
    
    # Scale positions to meters?
    # BigMac is usually 0.6mm isotropic.
    # 0.6mm = 0.0006 m
    voxel_size = 0.0006 
    positions = positions * voxel_size
    
    # Center the phantom
    center = jnp.mean(positions, axis=0)
    positions = positions - center
    
    # Properties
    T1_vals = T1_map[mask]
    T2_vals = T2_map[mask]
    
    # Downsample props
    if downsample > 1:
        T1_vals = T1_vals[::downsample]
        T2_vals = T2_vals[::downsample]
        
    # M0 - use T1 map intensity or uniform?
    # Real M0 map is usually PD. If not available, assume uniform or T1w intensity (masked).
    # Ideally should use quantitative M0/PD if in dataset.
    M0_vals = jnp.ones_like(T1_vals) 
    
    # Off-resonance
    off_resonance = jnp.zeros_like(T1_vals) 
    
    return IsochromatPhantom(
        positions=positions.astype(jnp.float32),
        T1=T1_vals.astype(jnp.float32),
        T2=T2_vals.astype(jnp.float32),
        M0=M0_vals.astype(jnp.float32),
        off_resonance=off_resonance.astype(jnp.float32)
    )
