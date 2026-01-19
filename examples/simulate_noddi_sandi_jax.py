
import jax
import jax.numpy as jnp
import numpy as np
from dmipy_jax.signal_models.cylinder_models import C1Stick
from dmipy_jax.signal_models.gaussian_models import G1Ball, G2Zeppelin
from dmipy_jax.signal_models.sphere_models import S1Sphere
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.acquisition import JaxAcquisition

def get_3shell_acquisition_scheme_jax():
    """
    Creates a simple 3-shell acquisition scheme for testing.
    """
    bvals = jnp.array([1000.0] * 30 + [2000.0] * 30 + [3000.0] * 30) * 1e6 # s/m^2
    
    # Random directions for demo
    key = jax.random.PRNGKey(42)
    dirs = jax.random.normal(key, (90, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Standard delta/Delta
    delta = 0.020
    Delta = 0.040
    
    return JaxAcquisition(bvalues=bvals, gradient_directions=dirs, delta=delta, Delta=Delta)

def simulate_noddi_like_dataset_jax(dimensions=(10, 10, 10)):
    """
    Simulates a NODDI-like dataset (Stick + Zeppelin + Ball) using dmipy_jax.
    """
    print("Constructing JAX NODDI model...")
    # 1. Define Components
    stick = C1Stick()
    zeppelin = G2Zeppelin()
    ball = G1Ball()
    
    # 2. Combine into Multi-Compartment Model
    noddi = JaxMultiCompartmentModel(models=[stick, zeppelin, ball])
    
    # 3. Generate Parameter Maps
    print(f"Generating parameter maps for {dimensions} volume...")
    
    # Grid
    x = jnp.linspace(0, 1, dimensions[0])
    y = jnp.linspace(0, 1, dimensions[1])
    z = jnp.linspace(0, 1, dimensions[2])
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # f_intra (Stick) - Linear gradient along X
    f_stick = 0.1 + 0.6 * X 
    
    # f_csf (Ball) - varying along Y
    f_csf = 0.5 * Y 
    
    # Normalize fractions
    total_non_zepp = f_stick + f_csf
    mask_overflow = total_non_zepp > 0.95
    
    # In JAX arrays are immutable, so we use where
    f_stick = jnp.where(mask_overflow, f_stick / (total_non_zepp / 0.95), f_stick)
    f_csf = jnp.where(mask_overflow, f_csf / (total_non_zepp / 0.95), f_csf)
    f_zeppelin = 1.0 - f_stick - f_csf
    
    # Parameters Dictionary
    parameters = {}
    parameters['partial_volume_0'] = f_stick
    parameters['partial_volume_1'] = f_zeppelin
    parameters['partial_volume_2'] = f_csf
    
    # Orientation
    # Rotate in XY plane along Z
    theta = Z * jnp.pi 
    mu_x = jnp.cos(theta)
    mu_y = jnp.sin(theta)
    mu_z = jnp.zeros_like(theta)
    
    mu = jnp.stack([mu_x, mu_y, mu_z], axis=-1) # (dim, dim, dim, 3)
    
    # Shared Orientation (Linking)
    parameters['C1Stick_1_mu'] = mu
    parameters['G2Zeppelin_1_mu'] = mu 
    
    # Diffusivities
    # Tortuosity constraint for Zeppelin:
    # lambda_perp = lambda_par * (1 - f_intra)
    lambda_par = 1.7e-9 # m^2/s
    
    # Broadcast scalar to volume shape if needed, or JAX handles scalar broadcasting naturally.
    # But for tortuosity calculation we might want explicit arrays effectively.
    
    parameters['C1Stick_1_lambda_par'] = jnp.full(dimensions, lambda_par)
    parameters['G2Zeppelin_1_lambda_par'] = jnp.full(dimensions, lambda_par)
    
    # Calculate Tortuosity
    lambda_perp = lambda_par * (1.0 - f_stick)
    parameters['G2Zeppelin_1_lambda_perp'] = lambda_perp
    
    parameters['G1Ball_1_lambda_iso'] = jnp.full(dimensions, 3.0e-9)
    
    # 4. Simulate
    print("Simulating signal with JAX...")
    scheme = get_3shell_acquisition_scheme_jax()
    
    # Convert dictionary to flat array for internal kernel if needed, 
    # but JaxMultiCompartmentModel usually has a simulate method? 
    # Checking `modeling_framework.py`: 
    # It has `model_func` which is the composted function taking (params_flat, acq).
    # And helper `parameter_dictionary_to_array`.
    
    params_flat = noddi.parameter_dictionary_to_array(parameters)
    
    # The `model_func` expects flat parameters.
    # And since we have spatial dimensions, we probably need to vmap if the model_func is for a single voxel?
    # Usually `compose_models` returns a function that expects (params, acq).
    # If params has extra dimensions (batch), does it auto-vectorize? 
    # The `JaxAcquisition` is a PyTree.
    
    # Let's inspect shapes.
    # params_flat shape: (N_params * N_voxels) ? No. 
    # `parameter_dictionary_to_array` concatenates.
    # If inputs are (10,10,10), output is (10,10,10, N_total_params).
    # We need to verify `parameter_dictionary_to_array` behavior for batching.
    
    # Looking at `modeling_framework.py`:
    # `jnp.concatenate([jnp.atleast_1d(p) for p in params_list])`
    # This might flatten everything into a huge 1D array if not careful, or concatenate along the first axis?
    # Wait, `jnp.atleast_1d` preserves dimensions?
    # If p is (10,10,10), `concatenate` on list of (10,10,10) will try to concat on axis 0 -> (N*10, 10, 10).
    # We want (10,10,10, N_params).
    
    # Actually `jax.vmap` is usually used for spatial dimensions.
    # The `OptimistixFitter` handles vmap. 
    # But for forward simulation, we should probably manually vmap or ensure expected shape.
    
    # Let's assume `model_func` is single-voxel.
    # We reshape parameters to (N_voxels, N_params)
    
    # Re-implementation of specialized to-array for this script to ensure correct shape:
    param_list_ordered = []
    # Order matters and must match `noddi.parameter_names`
    for name in noddi.parameter_names:
        val = parameters[name]
        if jnp.ndim(val) == 0:
            val = jnp.full(dimensions, val) # Broadcast scalars
        param_list_ordered.append(val)
        
    # Stack along last axis
    # Each val is (10,10,10) or (10,10,10, 3) for vectors?
    # Wait, `compose_models` usually expects flat parameter vector for a single voxel.
    # Vector parameters (mu) are usually flattened in the input to model_func?
    # e.g. [mu_x, mu_y, mu_z, lambda_par ...]
    
    # Correction: `model_func` created by `compose_models` takes a 1D array of parameters for one voxel.
    # So we need to construct (N_voxels, N_parameter_scalars).
    
    flat_param_maps = []
    for val in param_list_ordered:
        if val.shape[-1] == 3 and val.ndim > 1: # Vector field
             # Split into components
             flat_param_maps.append(val[..., 0])
             flat_param_maps.append(val[..., 1])
             flat_param_maps.append(val[..., 2])
        else:
             flat_param_maps.append(val)
             
    # Stack
    # dimensions + (N_params,)
    params_image = jnp.stack(flat_param_maps, axis=-1)
    
    # Flatten spatial dims for vmap
    params_flat_batch = params_image.reshape(-1, params_image.shape[-1])
    
    # Vmap the simulation
    # model_func(params, acq)
    model_vmapped = jax.vmap(noddi.model_func, in_axes=(0, None))
    
    signal_flat = model_vmapped(params_flat_batch, scheme)
    
    signal = signal_flat.reshape(dimensions + (-1,))
    
    print(f"JAX Signal simulated: {signal.shape}")
    return signal, scheme, parameters

if __name__ == "__main__":
    signal, scheme, params = simulate_noddi_like_dataset_jax()
    print("Done.")
