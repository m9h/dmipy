
import jax
import jax.numpy as jnp
import numpy as np
import os
from dmipy_jax.cylinder import C2Cylinder

def normalize_bvecs(bvecs):
    norms = jnp.linalg.norm(bvecs, axis=-1, keepdims=True)
    return jnp.where(norms == 0, 0, bvecs / norms)

def generate_gradient_table(bvals_shell, n_dirs_per_shell):
    """
    Generates a simple gradient table with multiple shells.
    """
    bvals = [0] # b0
    bvecs = [[0, 0, 0]]
    
    rng = np.random.default_rng(2024)
    
    for b in bvals_shell:
        # Puniform distribution on sphere
        indices = jnp.arange(0, n_dirs_per_shell, dtype=float) + 0.5
        phi = jnp.arccos(1 - 2 * indices / n_dirs_per_shell)
        theta = jnp.pi * (1 + 5**0.5) * indices

        x = jnp.cos(theta) * jnp.sin(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(phi)
        
        vecs = jnp.stack([x, y, z], axis=1)
        bvecs.append(np.array(vecs))
        bvals.extend([b] * n_dirs_per_shell)
        
    bvals = jnp.array(bvals)
    bvecs = jnp.concatenate([jnp.array(b) for b in bvecs], axis=0)
    
    return bvals, bvecs

def create_phantom_mask(shape):
    """
    Creates a simplified Shepp-Logan style phantom with 2 regions.
    """
    nx, ny, nz = shape
    grid_x, grid_y = jnp.meshgrid(jnp.linspace(-1, 1, ny), jnp.linspace(-1, 1, nx))
    
    # Region A: Central Ellipse (Radius 2um)
    mask_a = (grid_x**2 / 0.6**2 + grid_y**2 / 0.8**2) <= 1.0
    
    # Region B: Smaller inner ellipse/circle (Radius 5um) -> To distinguish, let's make it an inclusion
    # Actually, let's make Region A the main body, and Region B an internal structure.
    mask_b = ((grid_x - 0.2)**2 / 0.3**2 + (grid_y - 0.2)**2 / 0.3**2) <= 1.0
    
    # Resolve overlap: Region B overwrites Region A
    mask_a_final = mask_a & (~mask_b)
    mask_b_final = mask_b
    
    # Replicate across Z
    mask_a_vol = jnp.stack([mask_a_final] * nz, axis=-1)
    mask_b_vol = jnp.stack([mask_b_final] * nz, axis=-1)
    
    return mask_a_vol, mask_b_vol

def main():
    print("Generating Phantom...")
    
    # 1. Setup Acquisition
    bvals_list = [1000, 2000, 3000]
    n_dirs = 32
    bvals, bvecs = generate_gradient_table(bvals_list, n_dirs)
    
    print(f"Protocol: {len(bvals)} directions.")
    
    # 2. Setup Phantom Geometry
    shape = (50, 50, 5)
    mask_a, mask_b = create_phantom_mask(shape)
    
    # 3. Define Microstructure
    # Model: C2Cylinder
    # Parameters:
    # - lambda_par = 1.7e-9 m^2/s = 1.7 um^2/ms
    # - diameter: A=2um, B=5um -> Radius=1, 2.5 um. 
    #   Wait, prompt said Radius 2.0um and 5.0um. So Diameters = 4.0um and 10.0um.
    #   Convert to SI: 4e-6, 10e-6.
    # - Orientation: Z-axis for simplicity (theta=0).
    
    cyl_model = C2Cylinder()
    
    # Fixed parameters
    lambda_par = 1.7e-9 
    mu_a = jnp.array([0., 0.]) # Theta=0 (Z-axis)
    diam_a = 4.0e-6 # Radius 2um
    
    mu_b = jnp.array([jnp.pi/4, 0.]) # Theta=45 deg (tilted)
    diam_b = 10.0e-6 # Radius 5um
    
    # Pulse sequence params (Standard PGSE)
    big_delta = 30e-3 # 30ms
    small_delta = 10e-3 # 10ms
    
    # 4. Generate Signal
    # We can vectorize or just loop over pixels (for verification script, loop is fine, but JAX vmap is better)
    
    # Prepare parameter maps
    # Maps: (Nx, Ny, Nz, ...)
    p_diam = jnp.zeros(shape)
    p_diam = p_diam.at[mask_a].set(diam_a)
    p_diam = p_diam.at[mask_b].set(diam_b)
    
    p_theta = jnp.zeros(shape)
    p_theta = p_theta.at[mask_a].set(mu_a[0])
    p_theta = p_theta.at[mask_b].set(mu_b[0])
    
    p_phi = jnp.zeros(shape)
    p_phi = p_phi.at[mask_a].set(mu_a[1])
    p_phi = p_phi.at[mask_b].set(mu_b[1])
    
    # Flatten for computation
    N_vox = np.prod(shape)
    flat_diam = p_diam.ravel()
    flat_theta = p_theta.ravel()
    flat_phi = p_phi.ravel()
    
    # JAX vmap kernel
    def simulate_voxel(diam, theta, phi):
        # Return 0 if diameter is 0 (background)
        # Note: c2_cylinder might NaN on diameter=0
        # Safe guard:
        
        safe_diam = jnp.where(diam < 1e-9, 1e-6, diam) # dummy value
        
        mu = jnp.array([theta, phi])
        
        sig = cyl_model(
            bvals=bvals, 
            gradient_directions=bvecs,
            lambda_par=lambda_par,
            diameter=safe_diam,
            mu=mu,
            big_delta=big_delta,
            small_delta=small_delta
        )
        
        # Zero out background
        return jnp.where(diam < 1e-9, 0., sig)

    print("Simulating signals (JAX)...")
    pixel_kernel = jax.vmap(simulate_voxel)
    # Batch if memory issue? 50x50x5 = 12500 voxels. Tiny.
    
    signals_flat = pixel_kernel(flat_diam, flat_theta, flat_phi)
    signals = signals_flat.reshape(shape + (len(bvals),))
    
    # 5. Save output
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, 'phantom_signal.npy'), signals)
    np.save(os.path.join(data_dir, 'phantom_bvals.npy'), bvals)
    np.save(os.path.join(data_dir, 'phantom_bvecs.npy'), bvecs)
    
    # Save Metadata/Ground Truth
    ground_truth = {
        'diameters': p_diam,
        'mask_a': mask_a,
        'mask_b': mask_b,
        'regions': {
            'A': {'radius': 2.0, 'diameter': 4.0e-6},
            'B': {'radius': 5.0, 'diameter': 10.0e-6}
        },
        'big_delta': big_delta,
        'small_delta': small_delta
    }
    np.save(os.path.join(data_dir, 'phantom_ground_truth.npy'), ground_truth)
    
    print(f"Saved phantom data to {data_dir}")

if __name__ == "__main__":
    main()
