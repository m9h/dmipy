import jax
import jax.numpy as jnp
import numpy as np
import h5py
import argparse
from typing import Dict, Tuple

from dmipy_jax.acquisition import JaxAcquisition
from dmipy_jax.signal_models.cylinder_models import CallaghanRestrictedCylinder
from dmipy_jax.signal_models.zeppelin import Zeppelin
from dmipy_jax.signal_models.gaussian_models import Ball
from dmipy_jax.composer import compose_models

def get_acquisition(bvals_shell1=1000, bvals_shell2=2500, n_dirs=64,
                    delta=0.0129, Delta=0.0218):
    """
    Creates a standard 2-shell acquisition scheme.
    Timings are typical for HCP-like protocols.
    """
    # Shell 1
    bvals1 = jnp.ones(n_dirs) * bvals_shell1
    # Shell 2
    bvals2 = jnp.ones(n_dirs) * bvals_shell2
    
    # Random directions (just for simulation structure, in real training we might want specifically uniform)
    # Using simple fibonacci sphere or just random normal for this example to be quick
    key = jax.random.PRNGKey(0)
    dirs = jax.random.normal(key, (n_dirs * 2, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    
    # Add b0
    bvals_0 = jnp.zeros(1)
    dirs_0 = jnp.array([[1.0, 0.0, 0.0]])
    
    bvals = jnp.concatenate([bvals_0, bvals1, bvals2])
    bvecs = jnp.concatenate([dirs_0, dirs])
    
    return JaxAcquisition(
        bvalues=bvals,
        gradient_directions=bvecs,
        delta=delta,
        Delta=Delta
    )

def build_model():
    """
    Defines the 3-compartment tissue model.
    1. Intra-axonal: CallaghanRestrictedCylinder
    2. Extra-axonal: Zeppelin
    3. CSF: Ball
    """
    intra = CallaghanRestrictedCylinder()
    extra = Zeppelin()
    csf = Ball()
    
    model_func = compose_models([intra, extra, csf])
    # Parameter order in flattened array:
    # [Intra_Params..., Extra_Params..., CSF_Params..., f_intra, f_extra, f_csf]
    
    # Intra (Callaghan): mu(2), lambda_par(1), diameter(1), diff_perp(1) -> 5
    # Extra (Zeppelin): mu(2), lambda_par(1), lambda_perp(1) -> 4
    # CSF (Ball): lambda_iso(1) -> 1
    # Fractions: 3 (actually strictly typically N, but normalized they are N-1 DOF, here we pass N)
    
    # Total: 5 + 4 + 1 + 3 = 13 parameters
    return model_func

def sample_priors(key: jax.Array, n_samples: int) -> jnp.ndarray:
    """
    Samples model parameters from prior distributions.
    
    Returns:
        params: (n_samples, 13)
    """
    keys = jax.random.split(key, 10)
    
    # 1. Orientation (shared for Intra/Extra usually, but independent in full generic model)
    # Let's sample ONE orientation and share it to make it a coherent tissue model
    # Sample on sphere
    z = jax.random.uniform(keys[0], (n_samples,), minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(keys[1], (n_samples,), minval=0, maxval=2*jnp.pi)
    theta = jnp.arccos(z)
    
    # Intra-axonal parameters
    # mu: (theta, phi)
    # lambda_par: [0.5, 3.0] um^2/ms = [0.5, 3.0]e-9 m^2/s
    # diameter: [0.1, 10.0] um = [0.1, 10.0]e-6 m
    # diff_perp: usually < lambda_par. Let's say [0, 1.0]e-9? Or coupled?
    # For Callaghan, diff_perp is intra-cylindrical diffusion. Often small or fixed. 
    # Let's sample [0.1, 1.5]e-9.
    
    intra_lambda_par = jax.random.uniform(keys[2], (n_samples,), minval=0.5e-9, maxval=3.0e-9)
    intra_diameter = jax.random.uniform(keys[3], (n_samples,), minval=0.1e-6, maxval=10.0e-6)
    intra_diff_perp = jax.random.uniform(keys[4], (n_samples,), minval=0.1e-9, maxval=1.5e-9)
    # Enforce diff_perp <= lambda_par
    intra_diff_perp = jnp.minimum(intra_diff_perp, intra_lambda_par)

    # Extra-axonal parameters
    # mu: same as intra
    # lambda_par: same as intra (tortuosity assumption often says lambda_par_intra = lambda_par_extra)
    # Let's keep them slightly independent for "generic" learning but correlated is better for biology.
    # Let's assume lambda_par_extra = intra_lambda_par for this generator to be physically consistent.
    extra_lambda_par = intra_lambda_par
    
    # lambda_perp: [0.1, 2.0]e-9 (hindered)
    extra_lambda_perp = jax.random.uniform(keys[5], (n_samples,), minval=0.1e-9, maxval=2.0e-9)
    # Enforce extra_lambda_perp <= extra_lambda_par
    extra_lambda_perp = jnp.minimum(extra_lambda_perp, extra_lambda_par)

    # CSF parameters
    # lambda_iso: approx 3.0e-9 (free water at 37C)
    # Sample narrow range [2.8, 3.2]e-9
    csf_lambda = jax.random.uniform(keys[6], (n_samples,), minval=2.8e-9, maxval=3.2e-9)

    # Volume Fractions
    # Sample on simplex using Dirichlet
    # alpha = [1, 1, 1] (uniform on simplex)
    # fractional_concentrations (n_samples, 3)
    fractions = jax.random.dirichlet(keys[7], jnp.ones(3), shape=(n_samples,))
    
    # Pack parameters
    # [Intra (5), Extra (4), CSF (1), Fractions (3)]
    
    # Intra: mu(2), lambda_par, diameter, diff_perp
    mu_stack = jnp.stack([theta, phi], axis=1) # (N, 2)
    
    # Extra: mu(2), lambda_par, lambda_perp
    
    # CSF: lambda_iso
    
    # Concatenate
    # We need to being careful with shapes.
    
    _p = lambda x: x[:, None] if x.ndim == 1 else x
    
    params = jnp.concatenate([
        mu_stack, _p(intra_lambda_par), _p(intra_diameter), _p(intra_diff_perp), # Intra (5)
        mu_stack, _p(extra_lambda_par), _p(extra_lambda_perp),                   # Extra (4)
        _p(csf_lambda),                                                          # CSF (1)
        fractions                                                                # Fractions (3)
    ], axis=1)
    
    return params

def add_noise(key, signal, snr=30.0):
    """
    Adds Rician noise.
    """
    sigma = 1.0 / snr
    # Rician noise: sqrt( (S + n1)^2 + n2^2 )
    n1 = jax.random.normal(key, signal.shape) * sigma
    k2 = jax.random.split(key)[0]
    n2 = jax.random.normal(k2, signal.shape) * sigma
    
    noisy = jnp.sqrt( (signal + n1)**2 + n2**2 )
    return noisy

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dMRI training data.")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate.")
    parser.add_argument("--output", type=str, default="synthetic_data.h5", help="Output HDF5 filename.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Processing batch size.")
    args = parser.parse_args()

    print(f"Initializing model and acquisition...")
    acq = get_acquisition()
    # Move acq to GPU? handled by JAX default usually, but good to be explicit if needed.
    
    model = build_model()
    
    # JIT the batch simulator
    @jax.jit
    def simulate_batch(params, key):
        # params: (B, 13)
        # vmap over params, acq is broadcasted/fixed (closed over or passed)
        # Using in_axes=(0, None) for (params, acq)
        signals = jax.vmap(model, in_axes=(0, None))(params, acq)
        
        # Add noise
        noisy_signals = add_noise(key, signals, snr=30.0)
        return noisy_signals

    print(f"Generating {args.n_samples} samples...")
    
    # Open HDF5
    with h5py.File(args.output, 'w') as f:
        # Create datasets
        # Params: (N, 13)
        # Signals: (N, n_measurements)
        n_meas = len(acq.bvalues)
        dset_params = f.create_dataset("parameters", (args.n_samples, 13), dtype='float32')
        dset_signals = f.create_dataset("signals", (args.n_samples, n_meas), dtype='float32')
        dset_clean = f.create_dataset("signals_clean", (args.n_samples, n_meas), dtype='float32')
        
        # Loop in batches to save memory
        n_batches = int(np.ceil(args.n_samples / args.batch_size))
        
        key = jax.random.PRNGKey(42)
        
        for i in range(n_batches):
            print(f"Prototype Batch {i+1}/{n_batches}")
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, args.n_samples)
            current_batch_size = end_idx - start_idx
            
            key, subkey_p, subkey_n = jax.random.split(key, 3)
            
            # Sample params
            batch_params = sample_priors(subkey_p, current_batch_size)
            
            # Simulate
            # (Note: noise key needs to be split per batch or handled in function)
            batch_noisy = simulate_batch(batch_params, subkey_n)
            
            # Get clean signals for validation if needed (without noise function)
            # Re-running model without noise cost is small compared to correctness
            batch_clean = jax.vmap(model, in_axes=(0, None))(batch_params, acq)
            
            # Save to CPU -> Disk
            dset_params[start_idx:end_idx] = np.array(batch_params)
            dset_signals[start_idx:end_idx] = np.array(batch_noisy)
            dset_clean[start_idx:end_idx] = np.array(batch_clean)
            
    print(f"Done using dmipy-jax. Saved to {args.output}")

if __name__ == "__main__":
    main()
