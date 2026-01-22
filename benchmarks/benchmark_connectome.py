
import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx
from dmipy_jax.core.acquisition import acquisition_scheme_from_bvalues
from dmipy_jax.core.modeling_framework import JaxMultiCompartmentModel
from dmipy_jax.signal_models.cylinder_models import CallaghanRestrictedCylinder
from dmipy_jax.signal_models.gaussian_models import Ball

def run_benchmark():
    print("Loading Oracle Data...")
    try:
        data = np.load('data/connectome_oracle_pytorch.npz')
    except FileNotFoundError:
        print("Oracle data not found. Waiting for simulation to finish...")
        return

    signals_oracle = data['signals']
    bvals = data['bvals']
    bvecs = data['bvecs']
    mus = data['mu']
    diameters = data['diameter']
    f_intra = data['f_intra']
    
    n_samples = 1000 # Reduced for benchmarking
    f_intra = f_intra[:n_samples]
    diameters = diameters[:n_samples]
    mus = mus[:n_samples]
    print(f"Benchmarking on {n_samples} samples.")
    
    # 1. Setup Acquisition
    delta = 0.008
    Delta = 0.020
    acq = acquisition_scheme_from_bvalues(bvals, bvecs, delta=delta, Delta=Delta)
    
    # 2. Setup Model
    # Note: lambda_par in mm^2/s (matches bvals), diffusion_perp in m^2/s (matches diameter/Callaghan physics)
    lambda_par_val = 1.7e-3 # mm^2/s
    diffusion_perp_val = 1.7e-9 # m^2/s
    lambda_iso_val = 1.7e-3 # mm^2/s (Ball)
    
    # We construct the model. Parameters will be passed during call.
    # Instantiate with dummy values or None.
    cylinder = CallaghanRestrictedCylinder(diffusion_perpendicular=diffusion_perp_val)
    ball = Ball()
    
    model = JaxMultiCompartmentModel([cylinder, ball])
    
    # 3. Prepare Inputs
    # We need to constructing a parameter dictionary where keys match parameter names.
    # Names: CallaghanRestrictedCylinder_1_mu, ... but dependent on model naming.
    # Let's check parameter names.
    print(f"Model parameters: {model.parameter_names}")
    
    # Map input data to dictionary
    # mus: (N, 3). Convert to spherical? 
    # dmipy-jax models usually take Cartesian if shape is (3,) or spherical if (2,).
    # c3_cylinder_callaghan takes mu (3,) cartesian if provided.
    # BUT the Model class __call__ converts spherical to Cartesian!
    # "        # Convert spherical [theta, phi] to cartesian vector
    #          mu = jnp.asarray(mu) ..."
    # It implementation:
    # "        if mu.ndim > 0: theta=mu[0]; phi=mu[1] ... mu_cart = ..."
    # It assumes spherical input.
    # We should convert Cartesian mus to Spherical [theta, phi].
    
    # Conversion:
    # z = cos(theta) -> theta = arccos(z)
    # y = sin(theta)sin(phi), x = sin(theta)cos(phi) -> phi = arctan2(y, x)
    
    theta_in = np.arccos(mus[:, 2])
    phi_in = np.arctan2(mus[:, 1], mus[:, 0])
    # Stack [theta, phi]
    mu_sph = np.stack([theta_in, phi_in], axis=1) # (N, 2)
    
    # Use exact keys from model.parameter_names
    parameters = {
        'mu': jnp.array(mu_sph),
        'lambda_par': jnp.full((n_samples,), lambda_par_val),
        'diameter': jnp.array(diameters),
        'diffusion_perpendicular': jnp.full((n_samples,), diffusion_perp_val),
        'lambda_iso': jnp.full((n_samples,), lambda_iso_val),
        'partial_volume_0': jnp.array(f_intra),
        'partial_volume_1': 1.0 - jnp.array(f_intra) 
    }
    
    # 4. Run JAX Simulation
    print("JIT Compiling...")
    # Helper to run model on batch
    # Model call: model(parameters, acq)
    # But parameters are batched. model() isn't inherently vmapped over samples?
    # JaxMultiCompartmentModel calls models which might be vmapped?
    # Actually, usually we vmap the model call over parameters.
    
    @jax.jit
    def simulate_batch(params):
        # vmap over batch dimension (0)
        return jax.vmap(lambda p: model(p, acq))(params)
    
    # We need to transpose params to be struct of arrays or list of dicts?
    # vmap expects dictionary of arrays -> returns array of signals.
    # Our parameters dict contains arrays of shape (N, ...).
    # Correct.
    
    # Warmup
    # Take slice
    p_slice = {k: v[:100] for k, v in parameters.items()}
    _ = simulate_batch(p_slice)
    print("Compilation done.")
    
    print("Running Dmipy-Jax Simulation (GPU)...")
    start = time.time()
    
    # Run full batch (might need chunking if 100k is too big for GPU memory?
    # 100k * 200 floats = 20M floats = 80MB. Tiny.
    signals_jax = simulate_batch(parameters)
    # Wait for completion
    signals_jax.block_until_ready()
    
    duration = time.time() - start
    print(f"Dmipy-Jax Duration: {duration:.4f}s")
    print(f"Throughput: {n_samples/duration:.0f} samples/s")
    
    # Save
    np.savez('data/dmipy_jax_connectome.npz', signals=signals_jax)
    print("Saved 'data/dmipy_jax_connectome.npz'")

if __name__ == "__main__":
    run_benchmark()
