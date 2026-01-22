
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
# from jax.sharding import PositionalSharding

# Use dmipy-jax models
from dmipy_jax.signal_models import stick, gaussian_models
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
import dmipy.utils.spherical_mean
import numpy as np
import time

def get_acquisition_scheme(bval, n_dirs):
    """
    Generate an acquisition scheme with `n_dirs` encoded directions at `bval`.
    Also includes one b=0 image.
    """
    # Fibonacci Sphere (reused from numpy script or implement jax version if needed, 
    # but scheme generation is usually done on CPU once)
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
    
    # Return raw arrays for JAX usage
    return bvals, bvecs

def generate_phantom_params_numpy(shape):
    """
    Generate parameter maps on CPU (easier for masking logic).
    """
    Nx, Ny, Nz = shape
    
    # Initialize parameters
    mu1 = np.zeros(shape + (3,))
    mu2 = np.zeros(shape + (3,))
    f1 = np.zeros(shape)
    f2 = np.zeros(shape)
    f_iso = np.zeros(shape)
    
    # Define Regions
    mu1[..., 0] = 1.0 # X
    mu2[..., 1] = 1.0 # Y
    f_iso[:] = 1.0
    
    center_x, center_y = Nx // 2, Ny // 2
    width = Nx // 3
    x_coords = np.arange(Nx)
    y_coords = np.arange(Ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    mask1 = (np.abs(Y - center_y) < width // 2)
    mask2 = (np.abs(X - center_x) < width // 2)
    
    crossing_mask = mask1 & mask2
    single_1_mask = mask1 & (~mask2)
    single_2_mask = mask2 & (~mask1)
    
    f1[crossing_mask] = 0.3
    f2[crossing_mask] = 0.3
    f_iso[crossing_mask] = 0.4
    
    f1[single_1_mask] = 0.6
    f2[single_1_mask] = 0.0
    f_iso[single_1_mask] = 0.4
    
    f1[single_2_mask] = 0.0
    f2[single_2_mask] = 0.6
    f_iso[single_2_mask] = 0.4

    total = f1 + f2 + f_iso
    f1 /= total
    f2 /= total
    f_iso /= total
    
    return {
        'mu1': mu1, 'mu2': mu2,
        'f1': f1, 'f2': f2, 'f_iso': f_iso
    }

def vec2ang_jax(mu):
    # mu: (3,)
    x, y, z = mu[0], mu[1], mu[2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    r = jnp.maximum(r, 1e-12)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return jnp.array([theta, phi])

# JAX Model Definition
def simulate_voxel(mu1, mu2, f1, f2, f_iso, bvals, bvecs):
    """
    Simulates a single voxel signal using dmipy-jax models.
    """
    # Parameters
    lambda_par = 1.7e-3
    lambda_iso = 1.7e-3
    
    # Models
    # Note: dmipy_jax classes might be stateless or functional.
    # Looking at stick.py: class Stick -> __call__(bvals, dirs, **kwargs) -> c1_stick
    # We can use the functional interface or the class.
    
    # Stick 1
    mu1_ang = vec2ang_jax(mu1)
    s1 = stick.c1_stick(bvals, bvecs, mu1, lambda_par) # Optimization: pass cartesian directly if supported or converted inside
    # Wait, the reviewed stick.py used c1_stick and took mu as CARTESIAN in line 38 if passed?
    # No, line 38: c1_stick(..., mu_cart, ...). 
    # But __call__ takes spherical mu and converts it. 
    # Let's inspect c1_stick signature or Just use class to be safe? 
    # The class `Stick` takes spherical `mu` in `__call__` and converts to cartesian for `c1_stick`.
    # Let's try direct call to c1_stick if we have cartesian mu, avoiding double conversion.
    # But c1_stick might be imported from a kernel file? 
    # Let's stick (pun intended) to the class interface for now to be safe, or direct kernel.
    # Actually, let's use the Class which encapsulates usage.
    
    model_stick = stick.Stick()
    s1 = model_stick(bvals, bvecs, mu=mu1_ang, lambda_par=lambda_par)
    
    mu2_ang = vec2ang_jax(mu2)
    s2 = model_stick(bvals, bvecs, mu=mu2_ang, lambda_par=lambda_par)
    
    # Ball - assuming G1Ball exists in gaussian_models
    # Let's check gaussian_models or assume Ball exists.
    # If not, simple exp decay.
    s_iso = jnp.exp(-bvals * lambda_iso)
    
    # Combine
    signal = f1 * s1 + f2 * s2 + f_iso * s_iso
    return signal

# Vectorize over spatial dimensions (H, W, D)
# mu1: (H, W, D, 3) -> maps to (3,)
# f1: (H, W, D) -> maps to scalar
# bvals, bvecs: (N_meas,) -> Broadcasted (unmapped)
simulate_volume = jax.jit(jax.vmap(jax.vmap(jax.vmap(
    simulate_voxel, 
    in_axes=(0, 0, 0, 0, 0, None, None)), # D dimension
    in_axes=(0, 0, 0, 0, 0, None, None)), # W dimension
    in_axes=(0, 0, 0, 0, 0, None, None))  # H dimension
)

def main():
    print("Generating Synthetic Oracle Data (JAX Accelerated)...")
    
    # 1. Setup GPU Device
    try:
        print(f"JAX Devices: {jax.devices()}")
    except:
        print("No JAX devices found, running on CPU.")

    # 2. Protocol
    bval_hr = 3000.0
    n_dirs_hr = 90
    bvals_hr, bvecs_hr = get_acquisition_scheme(bval_hr, n_dirs_hr)
    
    # 3. Generating Params (CPU)
    hr_shape = (50, 50, 5)
    params = generate_phantom_params_numpy(hr_shape)
    
    # 4. Move to Device
    mu1 = jnp.array(params['mu1'])
    mu2 = jnp.array(params['mu2'])
    f1 = jnp.array(params['f1'])
    f2 = jnp.array(params['f2'])
    f_iso = jnp.array(params['f_iso'])
    
    bvals_jax = jnp.array(bvals_hr)
    bvecs_jax = jnp.array(bvecs_hr)
    
    # 5. Simulate
    print("Simulating High Resolution Signals...")
    start = time.time()
    hr_signal = simulate_volume(mu1, mu2, f1, f2, f_iso, bvals_jax, bvecs_jax)
    # Block until ready
    hr_signal.block_until_ready()
    print(f"Simulation took {time.time() - start:.4f}s")
    print(f"HR Signal Shape: {hr_signal.shape}") # (50, 50, 5, 91)
    
    # 6. Differentiable Downsampling
    print("Downsampling to Low Resolution...")
    # Target shape: 20x20x2
    # jax.image.resize expects (H, W, D, C)
    target_shape = (20, 20, 2, hr_signal.shape[-1])
    
    lr_signal_full = jax.image.resize(
        hr_signal, 
        shape=target_shape, 
        method='linear' # or 'cubic'
    )
    
    # 7. Subsample Directions
    # 90 -> 30
    dw_indices = jnp.linspace(1, 90, 30, dtype=int)
    keep_indices = jnp.concatenate((jnp.array([0]), dw_indices))
    
    lr_signal = lr_signal_full[..., keep_indices]
    lr_bvals = bvals_jax[keep_indices]
    lr_bvecs = bvecs_jax[keep_indices]
    
    print(f"LR Signal Shape: {lr_signal.shape}")
    
    # 8. Save
    print("Saving...")
    out_file = 'synthetic_oracle_data_jax.npz'
    np.savez_compressed(out_file, 
                        hr_signal=np.array(hr_signal),
                        hr_bvals=bvals_hr,
                        hr_bvecs=bvecs_hr,
                        gt_params=params, 
                        lr_signal=np.array(lr_signal),
                        lr_bvals=np.array(lr_bvals),
                        lr_bvecs=np.array(lr_bvecs),
                        hr_resolution=1.0,
                        lr_resolution=2.5)
    print("Done.")

if __name__ == "__main__":
    main()
